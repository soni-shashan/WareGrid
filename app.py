import streamlit as st
import pandas as pd
import plotly.express as px
import csv
import os
import datetime
import requests
import barcode
from barcode.writer import ImageWriter
import time
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from twilio.rest import Client
from huggingface_hub import InferenceClient

#-------------------Twilio Access Tokens-------------------------
account_sid = "your_account_sid"
auth_token = "your_auth_token"
from_whatsapp_number = "whatsapp:+your_from_whatsapp_number"
to_whatsapp_number = "whatsapp:+your_to_whatsapp_number"
#---------------------------------------------------------------


#------------------Hugging Face Token-----------------------------
model="Qwen/QwQ-32B-Preview"
token="your_hugging_face_token"
#---------------------------------------------------------------


# -----------------------------
#    File/Folder Configuration
# -----------------------------
DATA_FILE = "warehouse_data.csv"
TRANSACTION_FILE = "transactions.csv"
BARCODE_FOLDER = "barcodes"
CUSTOMER_FILE = "customers.csv"

# ------------- Twilio WhatsApp Notification -------------
def send_whatsapp_notification(low_stock_df):
    if low_stock_df.empty:
        return
    client = Client(account_sid, auth_token)
    body_lines = ["*Low Stock Alert!* The following items need restocking:\n"]
    for index, row in low_stock_df.iterrows():
        line = f"• Product ID: {row['Product ID']}, Name: {row['Name']}, Quantity: {row['Quantity']}"
        body_lines.append(line)
    message_body = "\n".join(body_lines)
    try:
        client.messages.create(body=message_body, from_=from_whatsapp_number, to=to_whatsapp_number)
        st.success("Low-stock WhatsApp notification sent successfully.")
    except Exception as e:
        st.error(f"Failed to send WhatsApp notification: {e}")

if not os.path.exists(BARCODE_FOLDER):
    os.makedirs(BARCODE_FOLDER)

# -----------------------------
#    Initialize CSV Structures
# -----------------------------
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Product ID", "Product Code", "Name", "Quantity", "Price", "Category"])

if not os.path.exists(TRANSACTION_FILE):
    with open(TRANSACTION_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Product ID", "Change in Quantity", "Reason", "Previous Quantity", "New Quantity"])

if not os.path.exists(CUSTOMER_FILE):
    with open(CUSTOMER_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Customer ID", "Name", "Email", "Phone"])

# -------------------------------
#       Core Data Functions
# -------------------------------
def load_data():
    """Loads warehouse data from the CSV file with string types enforced."""
    return pd.read_csv(DATA_FILE, dtype={'Product ID': str, 'Product Code': str, 'Name': str})

def save_data(df):
    """Saves the DataFrame back to the CSV file."""
    df.to_csv(DATA_FILE, index=False)

def log_transaction(product_id, quantity_change, reason, prev_qty, new_qty):
    """Appends a transaction record to the transactions.csv file."""
    with open(TRANSACTION_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            product_id,
            quantity_change,
            reason,
            prev_qty,
            new_qty
        ])

def load_transactions():
    """Loads transactions from CSV."""
    if os.path.exists(TRANSACTION_FILE):
        return pd.read_csv(TRANSACTION_FILE)
    return pd.DataFrame(columns=["Timestamp", "Product ID", "Change in Quantity", "Reason", "Previous Quantity", "New Quantity"])

def reconcile_inventory(df):
    """Checks for anomalies in quantity."""
    return df[(df['Quantity'] < 0) | (df['Quantity'] > 100000)]

def cycle_count(df):
    """Returns the first 5 products for cycle counting demonstration."""
    return df.head(5)

# ---------------------------------------
#    Barcode Generation / Printing
# ---------------------------------------
def generate_barcode_image(product_code):
    code128 = barcode.get_barcode_class('code128')
    code_object = code128(str(product_code), writer=ImageWriter())
    file_path = os.path.abspath(os.path.join(BARCODE_FOLDER, f"{product_code}"))
    code_object.save(file_path)
    png_path = file_path + ".png"
    if not os.path.exists(png_path):
        st.warning(f"Barcode file not found at {png_path} after saving.")
    return png_path

def print_barcode_image(file_path):
    if os.name == 'nt':
        try:
            os.startfile(file_path, "print")
        except Exception as e:
            st.warning(f"Printing failed: {e}")
    else:
        st.warning("Automatic printing is only demonstrated on Windows.")

# ---------------------------------------
#    Camera Barcode Scanner (WebRTC) - FIXED
# ---------------------------------------
if "camera_scanned_code" not in st.session_state:
    st.session_state.camera_scanned_code = None
if "scan_processed" not in st.session_state:
    st.session_state.scan_processed = False
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = 0

def barcode_scanner_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Enhanced barcode scanner with better detection and error handling"""
    try:
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocessing for better detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Upscale
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        decoded_objects = decode(gray)
        current_time = time.time()
        
        for obj in decoded_objects:
            if current_time - st.session_state.last_scan_time > 2:  # 2-second cooldown
                barcode_data = obj.data.decode("utf-8").strip()
                if barcode_data:
                    st.session_state.camera_scanned_code = barcode_data
                    st.session_state.last_scan_time = current_time
                    st.session_state.scan_processed = False
                    
                    # Draw bounding box on the original image
                    points = obj.polygon
                    if len(points) > 4:
                        hull = cv2.convexHull(np.array([point for point in points], dtype=np.int32))
                        cv2.polylines(img, [hull], True, (0, 255, 0), 3)
                    else:
                        n = len(points)
                        for j in range(n):
                            cv2.line(img, tuple(points[j]), tuple(points[(j+1) % n]), (0, 255, 0), 3)
                    break
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        st.error(f"Barcode processing error: {str(e)}")
        return frame

def camera_barcode_scan_interface(df):
    """Improved camera scanning interface with visual feedback"""
    st.subheader("📸 Real-time Barcode Scanner")
    
    with st.expander("Scanner Instructions", expanded=True):
        st.markdown("""
        1. Ensure good lighting on the barcode
        2. Hold steady 6-12 inches from camera
        3. Align barcode parallel to the screen
        4. Wait for auto-detection (green frame)
        """)
    
    # WebRTC Configuration
    webrtc_ctx = webrtc_streamer(
        key="barcode-scanner-v2",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=barcode_scanner_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    
    # Scan results handling
    if st.session_state.camera_scanned_code and not st.session_state.scan_processed:
        scanned_code = st.session_state.camera_scanned_code
        st.success(f"Detected Barcode: {scanned_code}")
        
        if scanned_code in df['Product Code'].values:
            product = df[df['Product Code'] == scanned_code].iloc[0]
            with st.form("scan_processing"):
                st.subheader("Process Inventory Change")
                quantity_change = st.number_input("Quantity Change", 
                                                min_value=1, 
                                                value=1,
                                                help="Enter the number of units to add/remove")
                direction = st.selectbox("Operation Type", 
                                        ["Check In", "Check Out"],
                                        help="Add stock or remove stock from inventory")
                
                if st.form_submit_button("Confirm Change"):
                    product_id = product['Product ID']
                    prev_qty = product['Quantity']
                    
                    if direction == "Check In":
                        new_qty = prev_qty + quantity_change
                        reason = "Camera Check-In"
                        qty_change = quantity_change
                    else:
                        new_qty = prev_qty - quantity_change
                        reason = "Camera Check-Out"
                        qty_change = -quantity_change
                    
                    if new_qty < 0:
                        st.error("Error: Negative inventory not allowed!")
                    else:
                        df.loc[df['Product Code'] == scanned_code, 'Quantity'] = new_qty
                        save_data(df)
                        log_transaction(product_id, qty_change, reason, prev_qty, new_qty)
                        st.success("Inventory updated successfully!")
                        st.session_state.scan_processed = True
                        st.experimental_rerun()
        else:
            st.error("Unrecognized barcode. Please add product first.")
            st.session_state.scan_processed = True

# ---------------------------------------
#    Plotly Visualizations
# ---------------------------------------
def plot_stock_levels(df):
    fig = px.bar(df, x='Name', y='Quantity', color='Category', title='Current Stock Levels by Product',
                 labels={'Name': 'Product Name', 'Quantity': 'Stock Quantity'}, hover_data=['Product Code', 'Price'])
    fig.update_layout(template='plotly_white', margin=dict(r=20, b=40, l=20, t=60), title_font=dict(size=24, family='Arial'))
    fig.update_traces(hovertemplate='<b>Product:</b> %{x}<br><b>Quantity:</b> %{y}<br><b>Price:</b> $%{customdata[1]}<br><b>Product Code:</b> %{customdata[0]}')
    return fig

def plot_category_distribution(df):
    fig = px.pie(df, names='Category', title='Product Distribution by Category', hole=0.3, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(template='plotly_white', margin=dict(r=20, b=40, l=20, t=60), title_font=dict(size=24, family='Arial'), legend_title='Category')
    fig.update_traces(textinfo='percent+label')
    return fig
def plot_sales_data(transaction_df, product_df):
    """Plots total quantity sold per product based on transaction logs."""
    # Filter sales transactions (negative quantity changes with "Sold" or "Sale to" reason)
    sales_data = transaction_df[
        (transaction_df['Change in Quantity'] < 0) & 
        ((transaction_df['Reason'] == 'Sold') | (transaction_df['Reason'].str.startswith('Sale to')))
    ].copy()
    
    if sales_data.empty:
        return None
    
    # Calculate total sold quantity per product (absolute value since negative indicates sold)
    sales_data['Sold Quantity'] = sales_data['Change in Quantity'].abs()
    sales_summary = sales_data.groupby('Product ID')['Sold Quantity'].sum().reset_index()
    
    # Merge with product data to get product names
    sales_summary = sales_summary.merge(product_df[['Product ID', 'Name']], on='Product ID', how='left')
    
    
    # Handle missing names by falling back to Product ID
    sales_summary['Name'] = sales_summary['Name'].fillna(sales_summary['Product ID'])
    
    # Ensure no negative or zero quantities (sanity check)
    if sales_summary['Sold Quantity'].eq(0).any() or sales_summary['Sold Quantity'].lt(0).any():
        st.warning("Warning: Invalid sold quantities detected. Check transaction data.")
        sales_summary = sales_summary[sales_summary['Sold Quantity'] > 0]
    
    # Create bar chart
    if sales_summary.empty:
        return None
    
    fig = px.bar(
        sales_summary,
        x='Name',
        y='Sold Quantity',
        title='Total Products Sold',
        labels={'Name': 'Product ID', 'Sold Quantity': 'Quantity Sold'},
        color='Name',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data=['Product ID']
    )
    fig.update_layout(
        template='plotly_white',
        margin=dict(r=20, b=40, l=20, t=60),
        title_font=dict(size=24, family='Arial'),
        showlegend=False
    )
    fig.update_traces(
        hovertemplate='<b>Product:</b> %{x}<br><b>Quantity Sold:</b> %{y}<br><b>Product ID:</b> %{customdata[0]}'
    )
    return fig

# ---------------------------------------
#    Customer Data Management
# ---------------------------------------
def load_customers():
    if os.path.exists(CUSTOMER_FILE):
        return pd.read_csv(CUSTOMER_FILE, dtype={'Customer ID': str, 'Name': str, 'Email': str, 'Phone': str})
    return pd.DataFrame(columns=["Customer ID", "Name", "Email", "Phone"])

def save_customers(df_customers):
    df_customers.to_csv(CUSTOMER_FILE, index=False)

# ---------------------------------------
#    Hugging Face Chatbot Integration
# ---------------------------------------
inference = InferenceClient(model=model, token=token)

def handle_chatbot_query(user_message: str, df: pd.DataFrame, max_retries=3, delay=5):
    system_prompt = (
        "You are a warehouse assistant. Use the following warehouse data to answer queries:\n"
        f"{df.to_string(index=False)}\n\n"
        "Provide concise, accurate answers based on this data when possible. For general questions or if the data doesn't suffice, "
        "respond naturally as a helpful assistant. If asked about stock levels, product details, or low stock, use the data directly."
    )
    full_prompt = f"{system_prompt}\n\nUser: {user_message}"
    user_message_lower = user_message.lower()

    if "stock" in user_message_lower or "quantity" in user_message_lower:
        for _, row in df.iterrows():
            if row['Name'].lower() in user_message_lower or row['Product ID'].lower() in user_message_lower:
                return f"The stock level for {row['Name']} (Product ID: {row['Product ID']}) is {row['Quantity']} units."
        return "I couldn't find that product. Please check the name or Product ID and try again."
    
    elif "low stock" in user_message_lower:
        low_stock = df[df['Quantity'] < 10]
        if not low_stock.empty:
            response = "Low stock items:\n" + "\n".join(
                f"- {row['Name']} (Product ID: {row['Product ID']}, Quantity: {row['Quantity']})"
                for _, row in low_stock.iterrows()
            )
            return response
        return "No items are currently low on stock."

    elif "price" in user_message_lower:
        for _, row in df.iterrows():
            if row['Name'].lower() in user_message_lower or row['Product ID'].lower() in user_message_lower:
                return f"The price of {row['Name']} (Product ID: {row['Product ID']}) is ${row['Price']}."
        return "I couldn't find that product. Please check the name or Product ID and try again."

    for attempt in range(max_retries):
        try:
            response = inference.text_generation(
                prompt=full_prompt,
                max_new_tokens=128,
                temperature=0.7,
                return_full_text=False
            )
            return response.strip()
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                st.warning(f"API unavailable, retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            return f"Sorry, I couldn’t connect to my knowledge base. How else can I assist you? (Error: {e})"

def chatbot_interface(df):
    st.subheader("Warehouse Chatbot (Powered by Hugging Face)")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for message in st.session_state["chat_history"]:
        speaker = "You" if message["role"] == "user" else "Bot"
        st.write(f"**{speaker}:** {message['content']}")
    user_input = st.text_input("Ask something about the warehouse (e.g., 'What’s the stock of X?', 'Any low stock items?')")
    if st.button("Send"):
        if user_input.strip():
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            response = handle_chatbot_query(user_input, df)
            st.session_state["chat_history"].append({"role": "bot", "content": response})

# ---------------------------------------
#    Main Streamlit Application
# ---------------------------------------
def main():
    st.set_page_config(page_title="Warehouse Management System", layout="wide")
    st.title("Dynamic WareGrid")
    df = load_data()
    df_customers = load_customers()

    with st.sidebar:
        st.header("Inventory & Customer Management")
        operation = st.selectbox(
            "Select Operation",
            ["Add Product", "Remove Product", "Modify Product", "Sell Product", "Barcode/RFID Scan", "Camera Barcode Scan", "Customer Management", "Chatbot"]
        )

        if operation == "Barcode/RFID Scan":
            st.subheader("Text-Based Barcode/RFID Scan")
            scanned_code = st.text_input("Enter or Scan Barcode/RFID").strip()
            quantity_change = st.number_input("Quantity Change", min_value=1, value=1)
            direction = st.selectbox("Operation Type", ["Check In", "Check Out"])
            if st.button("Process Scan"):
                if scanned_code in df['Product Code'].values:
                    product = df[df['Product Code'] == scanned_code].iloc[0]
                    product_id = product['Product ID']
                    prev_qty = product['Quantity']
                    if direction == "Check In":
                        new_qty = prev_qty + quantity_change
                        reason = "Barcode/RFID Check In"
                        qty_change_logged = quantity_change
                    else:
                        new_qty = prev_qty - quantity_change
                        reason = "Barcode/RFID Check Out"
                        qty_change_logged = -quantity_change
                    if new_qty < 0:
                        st.error("Operation would result in negative quantity. Cancelled.")
                    else:
                        df.loc[df['Product Code'] == scanned_code, 'Quantity'] = new_qty
                        save_data(df)
                        log_transaction(product_id, qty_change_logged, reason, prev_qty, new_qty)
                        st.success(f"Inventory updated for Product Code: {scanned_code}")
                else:
                    st.error("Scanned code not recognized. Please add the product first.")

        elif operation == "Camera Barcode Scan":
            camera_barcode_scan_interface(df)

        elif operation == "Add Product":
            with st.form("add_form"):
                st.subheader("Add New Product")
                product_id = st.text_input("Product ID")
                product_code = st.text_input("Product Code (for Barcode)")
                name = st.text_input("Product Name")
                quantity = st.number_input("Quantity", min_value=0, value=0)
                price = st.number_input("Price", min_value=0.0, value=0.0)
                category = st.selectbox("Category", ["Electronics", "Clothing", "Furniture", "Other"])
                if st.form_submit_button("Add Product"):
                    if not product_id.strip() or not name.strip() or not product_code.strip():
                        st.error("Product ID, Name, and Product Code cannot be empty.")
                    elif product_id in df['Product ID'].values:
                        st.error("Product ID already exists!")
                    elif product_code in df['Product Code'].values:
                        st.error("Product Code must be unique!")
                    else:
                        new_product = pd.DataFrame([[product_id, product_code, name, quantity, price, category]], columns=df.columns)
                        df = pd.concat([df, new_product], ignore_index=True)
                        save_data(df)
                        log_transaction(product_id, quantity, "Initial Stock", 0, quantity)
                        barcode_path = generate_barcode_image(product_code)
                        st.success("Product added successfully!")
                        st.image(barcode_path, caption=f"Barcode for {product_code}")
                        print_barcode_image(barcode_path)

        elif operation == "Remove Product":
            st.subheader("Remove Product")
            product_id = st.text_input("Enter Product ID to remove")
            if st.button("Remove"):
                if product_id in df['Product ID'].values:
                    product = df[df['Product ID'] == product_id].iloc[0]
                    prev_qty = product['Quantity']
                    df = df[df['Product ID'] != product_id]
                    save_data(df)
                    log_transaction(product_id, -prev_qty, "Product Removal", prev_qty, 0)
                    st.success("Product removed successfully!")
                else:
                    st.error("Product ID not found!")

        elif operation == "Modify Product":
            st.subheader("Modify Product")
            product_id = st.text_input("Enter Product ID to modify")
            if product_id in df['Product ID'].values:
                product = df[df['Product ID'] == product_id].iloc[0]
                with st.form("modify_form"):
                    new_quantity = st.number_input("Quantity", value=int(product['Quantity']), min_value=0)
                    new_price = st.number_input("Price", value=float(product['Price']), min_value=0.0)
                    categories_list = ["Electronics", "Clothing", "Furniture", "Other"]
                    idx = categories_list.index(product['Category']) if product['Category'] in categories_list else 3
                    new_category = st.selectbox("Category", categories_list, index=idx)
                    if st.form_submit_button("Update Product"):
                        prev_qty = product['Quantity']
                        df.loc[df['Product ID'] == product_id, ['Quantity', 'Price', 'Category']] = [new_quantity, new_price, new_category]
                        save_data(df)
                        quantity_change = new_quantity - prev_qty
                        log_transaction(product_id, quantity_change, "Manual Update", prev_qty, new_quantity)
                        st.success("Product updated successfully!")
            else:
                st.warning("Enter a valid Product ID")

        elif operation == "Sell Product":
            st.subheader("Sell a Product to a Customer")
            if df.empty or df_customers.empty:
                st.warning("No products or customers found. Please add them first.")
            else:
                customer_ids = df_customers["Customer ID"].tolist()
                selected_customer = st.selectbox("Select Customer", customer_ids)
                product_ids = df["Product ID"].tolist()
                selected_product_id = st.selectbox("Select Product ID", product_ids)
                sale_quantity = st.number_input("Quantity to Sell", min_value=1, value=1)
                if st.button("Confirm Sale"):
                    product_row = df[df["Product ID"] == selected_product_id].iloc[0]
                    prev_qty = product_row["Quantity"]
                    if sale_quantity > prev_qty:
                        st.error("Not enough quantity in stock to complete the sale.")
                    else:
                        new_qty = prev_qty - sale_quantity
                        reason = f"Sale to {selected_customer}"
                        df.loc[df["Product ID"] == selected_product_id, "Quantity"] = new_qty
                        save_data(df)
                        log_transaction(selected_product_id, -sale_quantity, reason, prev_qty, new_qty)
                        st.success(f"Successfully sold {sale_quantity} unit(s) of Product ID '{selected_product_id}' to Customer '{selected_customer}'.")

        elif operation == "Customer Management":
            st.subheader("Manage Customers")
            customer_action = st.selectbox("Select Customer Operation", ["Add Customer", "Remove Customer", "Modify Customer"])
            if customer_action == "Add Customer":
                with st.form("add_customer_form"):
                    cust_id = st.text_input("Customer ID")
                    cust_name = st.text_input("Name")
                    cust_email = st.text_input("Email")
                    cust_phone = st.text_input("Phone")
                    if st.form_submit_button("Add Customer"):
                        if not cust_id or not cust_name:
                            st.error("Customer ID and Name cannot be empty.")
                        elif cust_id in df_customers["Customer ID"].values:
                            st.error("Customer ID already exists!")
                        else:
                            new_customer = pd.DataFrame([[cust_id, cust_name, cust_email, cust_phone]], columns=["Customer ID", "Name", "Email", "Phone"])
                            df_customers = pd.concat([df_customers, new_customer], ignore_index=True)
                            save_customers(df_customers)
                            st.success("Customer added successfully!")
            elif customer_action == "Remove Customer":
                cust_id = st.text_input("Enter Customer ID to remove:")
                if st.button("Remove Customer"):
                    if cust_id in df_customers["Customer ID"].values:
                        df_customers = df_customers[df_customers["Customer ID"] != cust_id]
                        save_customers(df_customers)
                        st.success("Customer removed successfully!")
                    else:
                        st.error("Customer ID not found.")
            elif customer_action == "Modify Customer":
                cust_id = st.text_input("Enter Customer ID to modify:")
                if cust_id in df_customers["Customer ID"].values:
                    cust_record = df_customers[df_customers["Customer ID"] == cust_id].iloc[0]
                    with st.form("modify_customer_form"):
                        new_name = st.text_input("Name", value=cust_record["Name"])
                        new_email = st.text_input("Email", value=cust_record["Email"])
                        new_phone = st.text_input("Phone", value=cust_record["Phone"])
                        if st.form_submit_button("Update Customer"):
                            df_customers.loc[df_customers["Customer ID"] == cust_id, ["Name", "Email", "Phone"]] = [new_name, new_email, new_phone]
                            save_customers(df_customers)
                            st.success("Customer updated successfully!")
                else:
                    st.warning("Enter a valid Customer ID")
            st.subheader("Current Customers")
            st.dataframe(df_customers)

        elif operation == "Chatbot":
            chatbot_interface(df)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Current Inventory")
        if not df.empty:
            st.dataframe(df[['Product ID', 'Product Code', 'Name', 'Quantity', 'Price']], height=400)
        else:
            st.info("No products found. Please add products.")
        low_stock = df[df['Quantity'] < 10]
        if not low_stock.empty:
            st.subheader("⚠️ Low Stock Alert")
            st.dataframe(low_stock[['Product ID', 'Name', 'Quantity']])
            send_whatsapp_notification(low_stock)
        st.subheader("Product Search")
        search_term = st.text_input("Search by Name, Product ID, or Product Code").strip()
        if search_term:
            # Handle potential NaN values by filling with empty string
            search_result = df[
                df['Name'].fillna('').str.contains(search_term, case=False) |
                df['Product ID'].fillna('').str.contains(search_term, case=False) |
                df['Product Code'].fillna('').str.contains(search_term, case=False)
            ]
            st.dataframe(search_result)
        st.subheader("Price Sorting")
        sort_order = st.selectbox("Select sort order", ["Ascending", "Descending"])
        sorted_df = df.sort_values(by='Price', ascending=(sort_order == "Ascending"))
        st.dataframe(sorted_df[['Name', 'Price']])
        st.subheader("Reconciliation Check")
        anomalies = reconcile_inventory(df)
        if not anomalies.empty:
            st.warning("Potential anomalies detected:")
            st.dataframe(anomalies)
        st.subheader("Cycle Counting")
        if st.button("Select Items for Cycle Count"):
            to_count = cycle_count(df)
            st.dataframe(to_count)
        st.subheader("Transaction Logs")
        transaction_data = load_transactions()
        if not transaction_data.empty:
            st.dataframe(transaction_data)
        else:
            st.info("No transactions logged yet.")

    with col2:
        st.subheader("Visualizations")
        if not df.empty:
            st.plotly_chart(plot_stock_levels(df), use_container_width=True)
            st.plotly_chart(plot_category_distribution(df), use_container_width=True)
            
            # Add Sales Visualization
            transaction_data = load_transactions()
            if not transaction_data.empty:
                sales_fig = plot_sales_data(transaction_data, df)
                if sales_fig:
                    st.plotly_chart(sales_fig, use_container_width=True)
                else:
                    st.info("No sales transactions recorded yet.")
            else:
                st.warning("No transaction data available to display sales.")
        else:
            st.warning("No products to display in charts.")

if __name__ == "__main__":
    main()