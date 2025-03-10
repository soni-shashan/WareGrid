# Dynamic WareGrid - Warehouse Management System

**Dynamic WareGrid** is a modern, feature-rich warehouse management system built using Python and Streamlit. It provides real-time inventory tracking, barcode generation and scanning, customer management, transaction logging, data visualization, and WhatsApp notifications for low stock alerts. The system integrates advanced tools like Plotly for visualizations, OpenCV for barcode scanning, and a Hugging Face chatbot for natural language queries.

## Features

- **Inventory Management**: Add, remove, modify, and sell products with detailed tracking.
- **Barcode Support**: Generate barcodes, print them, and scan them via text input or webcam.
- **Customer Management**: Manage customer records and link sales to customers.
- **Real-Time Insights**: Visualize stock levels, category distribution, and sales data with Plotly charts.
- **Transaction Logging**: Record all inventory changes with timestamps and reasons.
- **Low Stock Alerts**: Automatically send WhatsApp notifications for items with low stock (<10 units).
- **Chatbot Assistance**: Query inventory data using a Hugging Face-powered natural language chatbot.
- **Reconciliation & Cycle Counting**: Detect anomalies and perform cycle counts for inventory accuracy.
- **Search & Sorting**: Search products by name, ID, or code, and sort by price.

## Demo

![image](https://github.com/user-attachments/assets/35ea4a4c-b459-4ad0-b41f-b323cf92f108)
![image](https://github.com/user-attachments/assets/81c07392-395a-4d7b-a0b9-e011fb232f4b)

*Caption: Real-time barcode scanning and inventory dashboard.*

## Prerequisites

- Python 3.8+
- A Twilio account for WhatsApp notifications (optional)
- A Hugging Face API token for chatbot functionality (optional)
- Webcam (for barcode scanning via camera)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/username/dynamic-waregrid.git
   cd dynamic-waregrid

2. **Set Up a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

- Open your browser to http://localhost:8501 to access the app.
### Usage
1. **Sidebar Operations:**
- Add, remove, or modify products and customers.
- Scan barcodes manually or use the webcam for real-time scanning.
- Sell products to registered customers.
- Interact with the chatbot for inventory queries.
2. **Main Dashboard:**
- View current inventory, low stock alerts, and transaction logs.
- Search products and sort by price.
- Perform reconciliation checks and cycle counts.
3. **Visualizations:**
- Analyze stock levels, category distribution, and sales data with interactive Plotly charts.
4. **Notifications:**
- Receive WhatsApp alerts for low stock items (requires Twilio setup).
5. **Chatbot:**
- Ask questions like "What’s the stock of X?" or "Any low stock items?" for instant responses.

### File Structure
  ```text
  WareGrid/
  ├── app.py              # Main Streamlit application
  ├── warehouse_data.csv  # Inventory data (auto-generated)
  ├── transactions.csv    # Transaction logs (auto-generated)
  ├── customers.csv       # Customer data (auto-generated)
  ├── barcodes/           # Folder for generated barcode images
  ├── requirements.txt    # Python dependencies
  └── README.md           # Project documentation
  ```

### Configuration
- **Twilio:** Replace account_sid, auth_token, from_whatsapp_number, and to_whatsapp_number in send_whatsapp_notification() with your credentials.
- **Hugging Face:** Update the InferenceClient token in handle_chatbot_query() with your Hugging Face API token.
- **Barcode Folder:** The barcodes/ directory is created automatically to store generated barcode images.
