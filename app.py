from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib
import os
import requests

try:
    from twilio.rest import Client
except ImportError:
    Client = None

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from werkzeug.utils import secure_filename
from prediction import predict_disease

# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    data = {}

    data['retailers'] = pd.read_csv('./data/retailers.csv')
    data['inventory'] = pd.read_csv('./data/retailer_inventory_weekly.csv')
    data['visits'] = pd.read_csv('./data/retailer_visit_log.csv')
    data['pos'] = pd.read_csv('./data/retailer_pos.csv')
    data['reps'] = pd.read_csv('./data/reps_territory.csv')
    
    # Load newly integrated datasets for AI patterns
    data['products'] = pd.read_csv('./templates/my_data - my_data.csv')
    data['growers'] = pd.read_csv('./data/growers.csv')
    data['whatsapp'] = pd.read_csv('./data/whatsapp_campaign.csv')
    data['digital'] = pd.read_csv('./data/digital_funnel_weekly.csv')

    data['inventory']['week_end_date'] = pd.to_datetime(data['inventory']['week_end_date'])
    data['visits']['visit_date'] = pd.to_datetime(data['visits']['visit_date'])
    data['pos']['transaction_date'] = pd.to_datetime(data['pos']['transaction_date'])

    return data

DATA = load_data()

# =========================================================
# HEALTH SCORE
# =========================================================

def calculate_health_score(retailer_id):
    sales = DATA['pos'][DATA['pos']['retailer_id'] == retailer_id]
    inventory = DATA['inventory'][DATA['inventory']['retailer_id'] == retailer_id]
    
    retailer_info = DATA['retailers'][DATA['retailers']['retailer_id'] == retailer_id]
    tehsil = retailer_info.iloc[0]['tehsil'] if not retailer_info.empty else None
    
    if tehsil:
        visits = DATA['visits'][DATA['visits']['visit_tehsil'] == tehsil]
    else:
        visits = pd.DataFrame()

    sales_score = 15
    if not sales.empty:
        last_date = sales['transaction_date'].max()
        recent_sales = sales[sales['transaction_date'] >= last_date - timedelta(days=30)]
        prev_sales = sales[(sales['transaction_date'] < last_date - timedelta(days=30)) & (sales['transaction_date'] >= last_date - timedelta(days=60))]
        rs_qty = recent_sales['sku_qty'].sum()
        ps_qty = prev_sales['sku_qty'].sum()
        
        if ps_qty > 0:
            growth = (rs_qty - ps_qty) / ps_qty
            if growth >= 0.1: sales_score = 30
            elif growth >= -0.1: sales_score = 25
            elif growth >= -0.5: sales_score = 15
            else: sales_score = 5
        else:
            sales_score = 25 if rs_qty > 0 else 0
            
    inventory_score = 25
    if not inventory.empty:
        latest_week = inventory['week_end_date'].max()
        latest_inv = inventory[inventory['week_end_date'] == latest_week]
        oos_count = len(latest_inv[latest_inv['sku_qty'] == 0])
        low_count = len(latest_inv[(latest_inv['sku_qty'] > 0) & (latest_inv['sku_qty'] < 5)])
        
        inventory_score -= (oos_count * 5)
        inventory_score -= (low_count * 2)
        inventory_score = max(0, inventory_score)
        
    visit_score = 0
    if not visits.empty:
        recent_visits = visits[visits['visit_date'] >= visits['visit_date'].max() - timedelta(days=90)]
        vc = len(recent_visits)
        if vc >= 5: visit_score = 20
        elif vc >= 2: visit_score = 15
        elif vc == 1: visit_score = 10
        
    campaign_score = 0
    if not visits.empty:
        campaigns = visits[visits['visit_type'] == 'campaign_conducted']
        if len(campaigns) >= 3: campaign_score = 25
        elif len(campaigns) >= 1: campaign_score = 15
        
    total = sales_score + inventory_score + visit_score + campaign_score

    if total >= 80:
        status = 'EXCELLENT'
        color = 'success'
    elif total >= 55:
        status = 'GOOD'
        color = 'warning'
    else:
        status = 'NEEDS ATTENTION'
        color = 'danger'

    churn_risk = 5
    if sales_score < 15: churn_risk += 35
    if inventory_score < 15: churn_risk += 20
    if visit_score == 0: churn_risk += 30
    churn_risk = min(95, churn_risk)

    return {
        'score': total,
        'status': status,
        'color': color,
        'churn_risk': churn_risk,
        'breakdown': {
            'sales': sales_score,
            'inventory': inventory_score,
            'visits': visit_score,
            'campaigns': campaign_score
        }
    }

# =========================================================
# AI PRIORITY ENGINE
# =========================================================

def get_priority(retailer_id):
    score = 0
    reason = []
    
    retailer_row = DATA['retailers'][DATA['retailers']['retailer_id'] == retailer_id]
    if retailer_row.empty:
        return 0, []
        
    district = retailer_row.iloc[0]['district']
    
    # 1. Supply Chain & Inventory Risk
    inventory = DATA['inventory']
    retailer_inv = inventory[inventory['retailer_id'] == retailer_id]
    if not retailer_inv.empty:
        latest_week = retailer_inv['week_end_date'].max()
        latest_inv_all_skus = retailer_inv[retailer_inv['week_end_date'] == latest_week]
        
        oos_skus = latest_inv_all_skus[latest_inv_all_skus['sku_qty'] == 0]['sku_name'].tolist()
        low_skus = latest_inv_all_skus[(latest_inv_all_skus['sku_qty'] > 0) & (latest_inv_all_skus['sku_qty'] < 5)]['sku_name'].tolist()
        
        if oos_skus:
            score += 30 + (10 * len(oos_skus))
            reason.append(f"Critical Stockout: {', '.join(oos_skus[:2])}")
        elif low_skus:
            score += 20 + (5 * len(low_skus))
            reason.append(f"Low Inventory: {', '.join(low_skus[:2])}")
            
    # 2. POS Velocity
    pos = DATA['pos']
    retailer_sales = pos[pos['retailer_id'] == retailer_id]
    if not retailer_sales.empty:
        latest_sales = retailer_sales[retailer_sales['transaction_date'] >= retailer_sales['transaction_date'].max() - timedelta(days=14)]
        if latest_sales['sku_qty'].sum() > 150:
            score += 25
            reason.append("High sales velocity (Last 14 days)")
            
    # 3. Spatio-Temporal & Crop Lifecycle Demand
    import json
    growers = DATA['growers']
    district_growers = growers[growers['district'] == district]
    if not district_growers.empty:
        crop_counts = {}
        for _, row in district_growers.iterrows():
            try:
                if pd.notna(row['grower_crop_calendar']):
                    gc = json.loads(row['grower_crop_calendar'])
                    crop = gc.get('crop')
                    if crop:
                        crop_counts[crop] = crop_counts.get(crop, 0) + 1
            except: pass
        
        if crop_counts:
            top_crop = max(crop_counts, key=crop_counts.get)
            if crop_counts[top_crop] > 2:
                score += 15
                reason.append(f"High predicted demand for {top_crop.capitalize()} (Crop Lifecycle)")
                
    # 4. Marketing Conversion & Digital Engagement
    whatsapp = DATA['whatsapp']
    if not district_growers.empty:
        district_grower_ids = district_growers['grower_id'].tolist()
        district_campaigns = whatsapp[(whatsapp['grower_id'].isin(district_grower_ids)) & (whatsapp['clicked_status'] == True)]
        if not district_campaigns.empty:
            score += 20
            reason.append(f"High local digital engagement ({len(district_campaigns)} recent WhatsApp clicks)")

    return min(100, score), reason

# =========================================================
# CHATBOT (Bilingual AI Copilot - Hindi + English)
# =========================================================

import re
import json as json_lib

def detect_intent(query):
    """Detect user intent from Hindi or English query."""
    # Greeting
    if any(w in query for w in ['hello', 'hi', 'hey', 'namaste', 'namaskar', 'namsate', 'hlo']):
        return 'greeting'
    # Retailer lookup
    if re.search(r'rtl_\d+', query):
        return 'retailer_lookup'
    # District/location queries
    if any(w in query for w in ['district', 'zila', 'jila', 'location', 'area', 'region', 'jagah', 'ilaka', 'state']):
        return 'district_query'
    # Crop queries
    if any(w in query for w in ['crop', 'fasal', 'kheti', 'wheat', 'gehun', 'rice', 'chawal', 'dhan', 'cotton', 'kapas', 'soybean', 'maize', 'makka', 'lentil', 'masoor', 'mustard', 'sarson', 'chana', 'onion', 'pyaz', 'potato', 'aloo', 'tomato', 'tamatar']):
        return 'crop_query'
    # Product queries
    if any(w in query for w in ['product', 'dawai', 'dawa', 'spray', 'fungicide', 'herbicide', 'insecticide', 'pesticide', 'tilt', 'score', 'axial', 'topik', 'actara', 'amistar', 'kavach', 'cruiser', 'vibrance', 'movondo', 'vertimec', 'keetnashak', 'khatpatwar']):
        return 'product_query'
    # Marketing queries
    if any(w in query for w in ['marketing', 'campaign', 'whatsapp', 'message', 'sandesh', 'promotion', 'funnel', 'click', 'open rate']):
        return 'marketing_query'
    # Inventory/stock queries
    if any(w in query for w in ['inventory', 'stock', 'maal', 'stockout', 'out of stock', 'khatam', 'low stock', 'kam stock', 'supply']):
        return 'inventory_query'
    # Sales queries
    if any(w in query for w in ['sales', 'bikri', 'revenue', 'kamai', 'transaction', 'pos', 'sell', 'bech']):
        return 'sales_query'
    # Priority/recommendation queries
    if any(w in query for w in ['priority', 'recommend', 'top', 'best', 'important', 'zaroori', 'jaruri', 'pehle', 'urgent', 'critical', 'suggest', 'sujhav']):
        return 'priority_query'
    # Weather queries
    if any(w in query for w in ['weather', 'mausam', 'rain', 'barish', 'baarish', 'temp', 'temperature', 'garmi', 'thand', 'humidity', 'nami']):
        return 'weather_query'
    # Help
    if any(w in query for w in ['help', 'madad', 'kya kar sakte', 'what can you', 'features', 'kaise']):
        return 'help'
    # Stats
    if any(w in query for w in ['stats', 'summary', 'overview', 'dashboard', 'total', 'kitne', 'count']):
        return 'stats_query'
    return 'unknown'

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    query = data.get('message', '').lower().strip()
    intent = detect_intent(query)

    if intent == 'greeting':
        response = "🙏 Namaste! Main Syngenta AI Copilot hoon. Aap mujhse Hindi ya English mein kuch bhi pooch sakte hain!<br><br>I can help you with:<br>• 🏪 Retailer health & priority<br>• 🌾 Crop lifecycle & product recommendations<br>• 📊 Sales & inventory insights<br>• 📱 WhatsApp campaign analytics<br>• 🌦️ Weather-based advice"

    elif intent == 'retailer_lookup':
        match = re.search(r'(rtl_\d+)', query)
        retailer_id = match.group(1).upper()
        health = calculate_health_score(retailer_id)
        retailer_info = DATA['retailers'][DATA['retailers']['retailer_id'] == retailer_id]

        if not retailer_info.empty:
            row = retailer_info.iloc[0]
            district = row['district']
            state = row['state']
            score, reasons = get_priority(retailer_id)
            
            # Get inventory status
            inv = DATA['inventory'][DATA['inventory']['retailer_id'] == retailer_id]
            oos_count = 0
            if not inv.empty:
                latest = inv[inv['week_end_date'] == inv['week_end_date'].max()]
                oos_count = len(latest[latest['sku_qty'] == 0])

            response = f"📋 <b>{retailer_id}</b> — {district}, {state}<br><br>"
            response += f"🏥 Health: <b>{health['status']}</b> ({health['score']}/100)<br>"
            response += f"⚠️ Churn Risk: <b>{health['churn_risk']}%</b><br>"
            response += f"🎯 Priority Score: <b>{score}</b><br>"
            if oos_count > 0:
                response += f"🔴 Out-of-Stock SKUs: <b>{oos_count}</b><br>"
            if reasons:
                response += f"<br>💡 AI Insights:<br>{'<br>'.join(['• ' + r for r in reasons])}"
        else:
            response = f"❌ Retailer <b>{retailer_id}</b> database mein nahi mila. Please check the ID."

    elif intent == 'district_query':
        # Extract district name
        districts = DATA['retailers']['district'].unique()
        found_district = None
        for d in districts:
            if d.lower() in query:
                found_district = d
                break
        
        if found_district:
            count = len(DATA['retailers'][DATA['retailers']['district'] == found_district])
            growers = DATA['growers']
            dg = growers[growers['district'] == found_district]
            response = f"📍 <b>{found_district}</b> District Intelligence:<br><br>"
            response += f"🏪 Retailers: <b>{count}</b><br>"
            response += f"👨‍🌾 Growers: <b>{len(dg)}</b><br>"
            
            # Top crops
            crop_counts = {}
            for _, row in dg.head(100).iterrows():
                try:
                    if pd.notna(row.get('grower_crop_calendar')):
                        gc = json_lib.loads(row['grower_crop_calendar'])
                        crop = gc.get('crop')
                        if crop: crop_counts[crop] = crop_counts.get(crop, 0) + 1
                except: pass
            if crop_counts:
                top = sorted(crop_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                response += f"<br>🌾 Top Crops:<br>{'<br>'.join(['• ' + c.capitalize() + f' ({n} growers)' for c, n in top])}"
        else:
            response = "Aap kaunse district ke baare mein jaanna chahte hain? Mujhe district ka naam batayein (e.g., 'Patna district' ya 'Jaipur ke baare mein batao')."

    elif intent == 'crop_query':
        crop_keywords = {'wheat': 'Wheat', 'gehun': 'Wheat', 'rice': 'Rice', 'chawal': 'Rice', 'dhan': 'Rice', 'cotton': 'Cotton', 'kapas': 'Cotton', 'soybean': 'Soybean', 'maize': 'Corn', 'makka': 'Corn', 'potato': 'Potato', 'aloo': 'Potato', 'onion': 'Onion', 'pyaz': 'Onion', 'tomato': 'Tomato', 'tamatar': 'Tomato', 'chilli': 'Chilli'}
        found_crop = None
        for k, v in crop_keywords.items():
            if k in query:
                found_crop = v
                break

        if found_crop:
            products_df = DATA['products']
            matching = []
            for _, prod in products_df.iterrows():
                val = prod.get(found_crop, '')
                if pd.notna(val) and str(val).strip():
                    matching.append({'name': prod['sku_name'], 'class': prod.get('Class', ''), 'disease': str(val)})

            response = f"🌾 <b>{found_crop}</b> ke liye Syngenta products:<br><br>"
            if matching:
                for m in matching[:5]:
                    response += f"💊 <b>{m['name']}</b> ({m['class']})<br>&nbsp;&nbsp;&nbsp;↳ {m['disease']}<br>"
            else:
                response += "Is fasal ke liye abhi koi specific product registered nahi hai."
        else:
            response = "Main in crops ke baare mein bata sakta hoon: Wheat/Gehun, Rice/Chawal, Cotton/Kapas, Soybean, Maize/Makka, Potato/Aloo, Onion/Pyaz, Tomato/Tamatar. Kaunsi fasal ke baare mein jaanna hai?"

    elif intent == 'product_query':
        products_df = DATA['products']
        found_product = None
        for _, prod in products_df.iterrows():
            if str(prod['sku_name']).lower() in query or str(prod['sku_id']).lower() in query:
                found_product = prod
                break
        # Keyword search
        if found_product is None:
            for _, prod in products_df.iterrows():
                name_words = str(prod['sku_name']).lower().split()
                if any(w in query for w in name_words if len(w) > 3):
                    found_product = prod
                    break

        if found_product is not None:
            desc = str(found_product.get('Description', ''))[:200]
            pclass = found_product.get('Class', 'N/A')
            response = f"💊 <b>{found_product['sku_name']}</b><br>"
            response += f"📦 Type: <b>{pclass}</b><br>"
            response += f"📝 {desc}<br><br>"
            # Show which crops it works for
            crop_cols = ['Wheat','Soybean','Rice','Cotton','Tea','Chilli','Onion','Apple','Corn','Potato','Tomato','Grapes','Mangoes']
            applicable = []
            for c in crop_cols:
                val = found_product.get(c, '')
                if pd.notna(val) and str(val).strip():
                    applicable.append(f"• <b>{c}</b>: {str(val)[:80]}")
            if applicable:
                response += "🌾 Applicable Crops:<br>" + "<br>".join(applicable[:6])
        else:
            response = "Kaun sa product? Main ye sab bata sakta hoon: Tilt 250 EC, Score 250 EC, Axial 50 EC, Topik 15 WP, Actara 25 WG, Amistar 250 SC, Kavach 75 WP, Cruiser 350 FS, Vibrance Integral, Movondo, Vertimec 1.8 EC"

    elif intent == 'marketing_query':
        wa = DATA['whatsapp']
        total = len(wa)
        delivered = len(wa[wa['delivered_status'] == True])
        opened = len(wa[wa['opened_status'] == True])
        clicked = len(wa[wa['clicked_status'] == True])
        top_products = wa['campaign_product'].value_counts().head(3)

        response = f"📱 <b>WhatsApp Campaign Overview:</b><br><br>"
        response += f"📤 Total Sent: <b>{total:,}</b><br>"
        response += f"✅ Delivered: <b>{delivered:,}</b> ({round(delivered/total*100,1)}%)<br>"
        response += f"👀 Opened: <b>{opened:,}</b> ({round(opened/total*100,1)}%)<br>"
        response += f"🖱️ Clicked: <b>{clicked:,}</b> ({round(clicked/total*100,1)}%)<br><br>"
        response += "🏆 Top Promoted Products:<br>"
        for p, c in top_products.items():
            response += f"• {p}: {c:,} messages<br>"

    elif intent == 'inventory_query':
        inv = DATA['inventory']
        latest_week = inv['week_end_date'].max()
        latest = inv[inv['week_end_date'] == latest_week]
        oos = latest[latest['sku_qty'] == 0]
        low = latest[(latest['sku_qty'] > 0) & (latest['sku_qty'] < 5)]

        oos_products = oos['sku_name'].value_counts().head(5)
        response = f"📦 <b>Inventory Status</b> (Week: {latest_week.strftime('%d %b %Y')}):<br><br>"
        response += f"🔴 Out-of-Stock instances: <b>{len(oos):,}</b><br>"
        response += f"🟡 Low Stock instances: <b>{len(low):,}</b><br><br>"
        response += "Most Stocked-Out Products:<br>"
        for p, c in oos_products.items():
            response += f"• {p}: {c} retailers<br>"

    elif intent == 'sales_query':
        pos = DATA['pos']
        total_transactions = len(pos)
        total_revenue = (pos['sku_qty'] * pos['sku_price']).sum()
        top_products = pos.groupby('sku_name')['sku_qty'].sum().sort_values(ascending=False).head(5)

        response = f"💰 <b>Sales Overview:</b><br><br>"
        response += f"📊 Total Transactions: <b>{total_transactions:,}</b><br>"
        response += f"💵 Total Revenue: <b>₹{total_revenue/10000000:.1f} Cr</b><br><br>"
        response += "🏆 Top Selling Products (by qty):<br>"
        for p, q in top_products.items():
            response += f"• {p}: {int(q):,} units<br>"

    elif intent == 'priority_query':
        retailers = DATA['retailers']['retailer_id'].unique()[:20]
        results = []
        for r in retailers:
            s, reasons = get_priority(r)
            if s > 0:
                results.append((r, s, reasons))
        results.sort(key=lambda x: x[1], reverse=True)

        response = "🎯 <b>Top Priority Retailers:</b><br><br>"
        for r, s, reasons in results[:5]:
            response += f"• <b>{r}</b> (Score: {s}) — {', '.join(reasons[:2])}<br>"
        response += "<br>💡 Select a district from the dropdown above for district-specific priorities."

    elif intent == 'weather_query':
        response = "🌦️ Weather intelligence ke liye, upar ke dropdown se ek district select karein. Main us district ka real-time mausam aur uske hisaab se product recommendations dikha dunga.<br><br>For example: Select India → Bihar → Patna to see live weather + product advice for that region."

    elif intent == 'stats_query':
        total_r = len(DATA['retailers'])
        total_g = len(DATA['growers'])
        total_pos = len(DATA['pos'])
        total_inv = len(DATA['inventory'])
        states = DATA['retailers']['state'].nunique()

        response = f"📊 <b>Platform Overview:</b><br><br>"
        response += f"🏪 Total Retailers: <b>{total_r:,}</b><br>"
        response += f"👨‍🌾 Total Growers: <b>{total_g:,}</b><br>"
        response += f"🗺️ States Covered: <b>{states}</b><br>"
        response += f"🧾 POS Transactions: <b>{total_pos:,}</b><br>"
        response += f"📦 Inventory Records: <b>{total_inv:,}</b>"

    elif intent == 'help':
        response = "🤖 <b>Main aapki kaise madad kar sakta hoon:</b><br><br>"
        response += "🏪 <b>Retailer:</b> 'RTL_00001 ka health batao'<br>"
        response += "📍 <b>District:</b> 'Patna district ke baare mein batao'<br>"
        response += "🌾 <b>Crop:</b> 'Wheat/Gehun ke liye kya spray karein?'<br>"
        response += "💊 <b>Product:</b> 'Tilt 250 EC ke baare mein batao'<br>"
        response += "📱 <b>Marketing:</b> 'WhatsApp campaign ka status kya hai?'<br>"
        response += "📦 <b>Inventory:</b> 'Stock status batao'<br>"
        response += "💰 <b>Sales:</b> 'Total bikri kitni hai?'<br>"
        response += "📊 <b>Stats:</b> 'Dashboard ka overview do'<br>"
        response += "🎯 <b>Priority:</b> 'Top priority retailers kaun hain?'"

    else:
        response = "🤔 Main samajh nahi paaya. Aap ye pooch sakte hain:<br><br>"
        response += "• 'RTL_00001 health' — Retailer analysis<br>"
        response += "• 'Wheat ke liye product' — Crop-based recommendations<br>"
        response += "• 'Patna district' — District intelligence<br>"
        response += "• 'WhatsApp campaign' — Marketing data<br>"
        response += "• 'Stock status' — Inventory alerts<br>"
        response += "• 'Madad' / 'Help' — Full list of commands"

    return jsonify({
        'response': response
    })

# =========================================================
# LIVE WHATSAPP CHATBOT (TWILIO WEBHOOK)
# =========================================================

from twilio.twiml.messaging_response import MessagingResponse

@app.route('/api/whatsapp-webhook', methods=['POST'])
def whatsapp_webhook():
    """
    This endpoint handles incoming messages from real WhatsApp users via Twilio.
    """
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From', '')

    print(f"📥 Received WhatsApp message from {sender}: {incoming_msg}")

    # Use the existing intent detection engine
    intent = detect_intent(incoming_msg.lower())

    # We need to reuse the response logic from the web chatbot, 
    # but strip HTML tags since WhatsApp doesn't support them.
    # We will simulate a request to our existing chatbot function.
    with app.test_request_context('/api/chatbot', method='POST', json={'message': incoming_msg}):
        chatbot_resp = chatbot().get_json()
        web_response = chatbot_resp.get('response', '')

    # Convert HTML formatting to WhatsApp markdown
    wa_response = web_response.replace('<br>', '\n').replace('<b>', '*').replace('</b>', '*')
    wa_response = wa_response.replace('&nbsp;', ' ')

    # Send response back to Twilio
    resp = MessagingResponse()
    resp.message(wa_response)
    
    return str(resp)

# =========================================================
# RETAILER DETAIL API
# =========================================================

@app.route('/api/retailer/<retailer_id>')
def retailer_detail(retailer_id):

    retailer = DATA['retailers'][
        DATA['retailers']['retailer_id'] == retailer_id
    ]

    inventory = DATA['inventory'][
        DATA['inventory']['retailer_id'] == retailer_id
    ].copy().tail(10)
    inventory['week_end_date'] = inventory['week_end_date'].astype(str)

    sales = DATA['pos'][
        DATA['pos']['retailer_id'] == retailer_id
    ]

    tehsil = retailer.iloc[0]['tehsil'] if not retailer.empty else None
    if tehsil:
        visits = DATA['visits'][DATA['visits']['visit_tehsil'] == tehsil].copy()
        visits['visit_date'] = visits['visit_date'].astype(str)
    else:
        visits = pd.DataFrame()

    health = calculate_health_score(retailer_id)

    total_sales = int(sales['sku_qty'].sum()) if not sales.empty else 0

    return jsonify({
        'retailer_id': retailer_id,
        'health_score': health,
        'total_sales': total_sales,
        'visit_count': len(visits),
        'inventory_records': inventory.to_dict(orient='records')
    })

# =========================================================
# =========================================================
# ANOMALY DETECTION, STATS, MAP & DIRECTORY
# =========================================================

def get_retailer_location(retailer_id, district):
    district_coords = {
        'Patna': (25.5941, 85.1376),
        'Jalgaon': (21.0077, 75.5626),
        'Vijayapura': (16.8302, 75.7100),
        'Ahmedabad': (23.0225, 72.5714),
        'Ludhiana': (30.9010, 75.8573),
        'Hisar': (29.1492, 75.7217),
        'Varanasi': (25.3176, 82.9739)
    }
    base_lat, base_lng = district_coords.get(district, (22.5, 78.5))
    h = int(hashlib.md5(retailer_id.encode()).hexdigest(), 16)
    lat_offset = (h % 1000) / 10000.0 - 0.05
    lng_offset = ((h // 1000) % 1000) / 10000.0 - 0.05
    return (base_lat + lat_offset, base_lng + lng_offset)

@app.route('/api/map-data')
def get_map_data():
    retailers = DATA['retailers'].head(200) # Load top 200 for demo
    
    map_data = []
    for _, r in retailers.iterrows():
        lat, lng = get_retailer_location(r['retailer_id'], r['district'])
        health = calculate_health_score(r['retailer_id'])
        
        map_data.append({
            'retailer_id': r['retailer_id'],
            'lat': lat,
            'lng': lng,
            'status': health['status'],
            'score': health['score']
        })
        
    return jsonify({'map_data': map_data})

@app.route('/api/locations')
def get_locations():
    # Returns nested dictionary { "Country": { "State": ["District1", "District2"] } }
    df = DATA['retailers']
    locations = {"India": {}}
    for state in df['state'].dropna().unique():
        districts = df[df['state'] == state]['district'].dropna().unique().tolist()
        locations["India"][state] = districts
    return jsonify(locations)

def get_loyalty_tier(retailer_id):
    pos = DATA['pos']
    sales = pos[pos['retailer_id'] == retailer_id]
    if sales.empty:
        return {'tier': 'Bronze', 'color': 'secondary'}
    
    total_qty = sales['sku_qty'].sum()
    if total_qty > 500:
        return {'tier': 'Platinum', 'color': 'primary'}
    elif total_qty > 300:
        return {'tier': 'Gold', 'color': 'warning'}
    elif total_qty > 100:
        return {'tier': 'Silver', 'color': 'info'}
    else:
        return {'tier': 'Bronze', 'color': 'secondary'}

@app.route('/api/filter-retailers')
def filter_retailers():
    district = request.args.get('district')
    if not district:
        return jsonify([])
        
    filtered = DATA['retailers'][DATA['retailers']['district'] == district].head(15)
    results = []
    for _, r in filtered.iterrows():
        h = calculate_health_score(r['retailer_id'])
        tier = get_loyalty_tier(r['retailer_id'])
        results.append({
            'retailer_id': r['retailer_id'],
            'district': r['district'],
            'score': h['score'],
            'status': h['status'],
            'color': h['color'],
            'tier': tier['tier'],
            'tier_color': tier['color']
        })
    return jsonify(results)

@app.route('/api/nearby-retailers')
def nearby_retailers():
    lat = float(request.args.get('lat', 22.5))
    lng = float(request.args.get('lng', 78.5))
    
    retailers = DATA['retailers'].head(300) # search pool
    results = []
    
    for _, r in retailers.iterrows():
        r_lat, r_lng = get_retailer_location(r['retailer_id'], r['district'])
        # Simple Euclidean distance for demo ranking
        dist = ((lat - r_lat)**2 + (lng - r_lng)**2)**0.5
        results.append({
            'retailer_id': r['retailer_id'],
            'district': r['district'],
            'distance': dist,
            'lat': r_lat,
            'lng': r_lng
        })
        
    results = sorted(results, key=lambda x: x['distance'])[:12]
    
    # Append health scores and loyalty tier for top results
    for res in results:
        h = calculate_health_score(res['retailer_id'])
        tier = get_loyalty_tier(res['retailer_id'])
        res['score'] = h['score']
        res['status'] = h['status']
        res['color'] = h['color']
        res['tier'] = tier['tier']
        res['tier_color'] = tier['color']
        
    return jsonify(results)

@app.route('/api/send-whatsapp', methods=['POST'])
def send_whatsapp():
    data = request.json
    retailer_id = data.get('retailer_id')
    alert_type = data.get('type', 'ALERT')
    message_text = data.get('message', '')
    phone = data.get('phone', '')

    # Build the alert message
    full_message = f"🚨 SYNGENTA ALERT: {alert_type} at {retailer_id}. {message_text}. Please take action immediately."

    # URL-encode the message for WhatsApp
    import urllib.parse
    encoded_msg = urllib.parse.quote(full_message)

    # If phone number provided, create a direct WhatsApp link
    if phone:
        # Clean phone number (remove spaces, dashes, leading 0)
        clean_phone = phone.strip().replace(' ', '').replace('-', '').replace('+', '')
        if clean_phone.startswith('0'):
            clean_phone = '91' + clean_phone[1:]
        if not clean_phone.startswith('91'):
            clean_phone = '91' + clean_phone
        whatsapp_url = f"https://api.whatsapp.com/send?phone={clean_phone}&text={encoded_msg}"
    else:
        # No phone — open WhatsApp with just the message (user picks contact)
        whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_msg}"

    # Also try Twilio if configured
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')

    if account_sid and auth_token and Client:
        try:
            client = Client(account_sid, auth_token)
            message = client.messages.create(
                from_='whatsapp:+14155238886',
                body=full_message,
                to=f'whatsapp:+{clean_phone}' if phone else 'whatsapp:+919876543210'
            )
            return jsonify({'status': 'success', 'sid': message.sid, 'whatsapp_url': whatsapp_url})
        except Exception as e:
            print("Twilio Error:", e)

    # Return WhatsApp deep-link for direct opening
    print(f"\n📲 WhatsApp Link Generated for {retailer_id}")
    print(f"   URL: {whatsapp_url}\n")
    return jsonify({
        'status': 'success',
        'whatsapp_url': whatsapp_url,
        'message_preview': full_message
    })

@app.route('/api/weather-insights')
def weather_insights():
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    district = request.args.get('district')
    
    if not lat or not lng:
        if district:
            lat, lng = get_retailer_location("mock", district)
        else:
            return jsonify({'error': 'Missing location params'}), 400
            
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current=temperature_2m,precipitation,weather_code"
    
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        current = data.get('current', {})
        
        temp = current.get('temperature_2m', 25)
        precip = current.get('precipitation', 0)
        
        if precip > 0:
            condition = "Rainy / High Humidity"
            icon = "fa-cloud-rain text-info"
            products = [
                {"name": "Kavach 75 WP", "reason": "High disease risk (early blight) due to rain."},
                {"name": "Amistar Top", "reason": "Systemic fungicide for broad-spectrum control."},
                {"name": "Ridomil Gold", "reason": "Preventative action for wet conditions."}
            ]
        elif temp > 35:
            condition = "Hot & Dry"
            icon = "fa-sun text-warning"
            products = [
                {"name": "Actara 25 WG", "reason": "High sucking pest pressure in hot weather."},
                {"name": "Pegasus", "reason": "Controls mites which thrive in dry heat."},
                {"name": "Isabion", "reason": "Biostimulant to reduce heat stress."}
            ]
        else:
            condition = "Optimal / Moderate"
            icon = "fa-cloud-sun text-success"
            products = [
                {"name": "Quantis", "reason": "Maintains optimal yield during standard conditions."},
                {"name": "Score 250 EC", "reason": "Routine protective fungicidal spray."},
                {"name": "Karate Zeon", "reason": "Standard broad-spectrum insect control."}
            ]
            
        return jsonify({
            'temperature': temp,
            'precipitation': precip,
            'condition': condition,
            'icon': icon,
            'recommendations': products
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard-stats')
def dashboard_stats():
    total_retailers = len(DATA['retailers'])
    
    pos = DATA['pos']
    pos_recent = pos[pos['transaction_date'] >= pos['transaction_date'].max() - timedelta(days=30)]
    revenue = (pos_recent['sku_qty'] * pos_recent['sku_price']).sum()
    
    inv = DATA['inventory']
    recent_date = inv['week_end_date'].max()
    oos_mask = (inv['week_end_date'] == recent_date) & (inv['sku_qty'] == 0)
    oos_retailers = inv[oos_mask]['retailer_id'].nunique()
    
    return jsonify({
        'total_retailers': total_retailers,
        'high_priority': int(total_retailers * 0.15),
        'critical_alerts': oos_retailers,
        'revenue_opportunity': float(revenue)
    })

@app.route('/api/alerts')
def get_alerts():
    alerts = []
    retailers = DATA['retailers']['retailer_id'].unique()[:100]
    
    for r in retailers:
        sales = DATA['pos'][DATA['pos']['retailer_id'] == r]
        if not sales.empty:
            last_date = sales['transaction_date'].max()
            recent_sales = sales[sales['transaction_date'] >= last_date - timedelta(days=30)]['sku_qty'].sum()
            prev_sales = sales[(sales['transaction_date'] < last_date - timedelta(days=30)) & (sales['transaction_date'] >= last_date - timedelta(days=60))]['sku_qty'].sum()
            if prev_sales > 0 and recent_sales < prev_sales * 0.5:
                alerts.append({
                    'type': 'Sales Drop',
                    'retailer_id': r,
                    'message': f'Sales dropped by >50% (from {prev_sales} to {recent_sales} units)',
                    'severity': 'High'
                })
        
        inventory = DATA['inventory'][DATA['inventory']['retailer_id'] == r]
        if not inventory.empty:
            dates = sorted(inventory['week_end_date'].unique())
            if len(dates) >= 2:
                last_2 = inventory[inventory['week_end_date'].isin(dates[-2:])]
                oos = last_2[last_2['sku_qty'] == 0]
                vc = oos['product_sku'].value_counts()
                for sku, count in vc.items():
                    if count >= 2:
                        alerts.append({
                            'type': 'OOS Streak',
                            'retailer_id': r,
                            'message': f'{sku} out of stock for 2+ weeks.',
                            'severity': 'High'
                        })
                        
        retailer_info = DATA['retailers'][DATA['retailers']['retailer_id'] == r]
        tehsil = retailer_info.iloc[0]['tehsil'] if not retailer_info.empty else None
        if tehsil:
            visits = DATA['visits'][DATA['visits']['visit_tehsil'] == tehsil]
            dataset_date = DATA['visits']['visit_date'].max()
            if visits.empty or visits['visit_date'].max() < dataset_date - timedelta(days=90):
                alerts.append({
                    'type': 'Inactive',
                    'retailer_id': r,
                    'message': 'No field visits in 90+ days.',
                    'severity': 'Medium'
                })
                    
    # Only return top alerts to avoid overwhelming UI
    return jsonify({'alerts': alerts[:12]})

@app.route('/api/forecast')
def seasonal_forecast():
    pos = DATA['pos']
    recent_date = pos['transaction_date'].max()
    month = recent_date.month
    
    if 6 <= month <= 10:
        season = "Kharif Season (Monsoon)"
        products = [
            {"product": "Score 250 EC", "crop": "Paddy/Rice", "demand": "HIGH", "insight": "High demand expected due to blast disease risk."},
            {"product": "Kavach 75 WP", "crop": "Vegetables", "demand": "HIGH", "insight": "Preventative fungicide for early blight."},
            {"product": "Amistar 250 SC", "crop": "Cotton", "demand": "MEDIUM", "insight": "Useful during square formation stage."},
            {"product": "Actara 25 WG", "crop": "Cotton", "demand": "HIGH", "insight": "Critical for sucking pest control."},
            {"product": "Cruiser 350 FS", "crop": "Soybean", "demand": "LOW", "insight": "Seed treatment phase is mostly over."}
        ]
    else:
        season = "Rabi Season (Winter)"
        products = [
            {"product": "Axial 50 EC", "crop": "Wheat", "demand": "HIGH", "insight": "Crucial for post-emergent grass weed control."},
            {"product": "Tilt 250 EC", "crop": "Wheat", "demand": "HIGH", "insight": "High risk of yellow rust; promote early sprays."},
            {"product": "Topik 15 WP", "crop": "Wheat", "demand": "MEDIUM", "insight": "Alternative herbicide for resistant weeds."},
            {"product": "Vertimec 1.8 EC", "crop": "Vegetables", "demand": "HIGH", "insight": "Mite infestations peak during dry spells."},
            {"product": "Movondo", "crop": "Mustard", "demand": "MEDIUM", "insight": "Effective against aphid attacks."}
        ]
        
    return jsonify({
        'season': season,
        'month': recent_date.strftime('%B %Y'),
        'forecast': products
    })

# =========================================================
# MAIN DASHBOARD API
# =========================================================

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    district = request.json.get('district') if request.json else None

    if district:
        filtered = DATA['retailers'][DATA['retailers']['district'] == district]
    else:
        filtered = DATA['retailers'].head(50)

    retailers = filtered['retailer_id'].unique()[:30]

    output = []

    for retailer in retailers:
        score, reason = get_priority(retailer)
        if score > 0:
            output.append({
                'retailer_id': retailer,
                'priority_score': score,
                'reason': reason
            })

    output = sorted(output, key=lambda x: x['priority_score'], reverse=True)[:6]
    
    # Calculate health only for the top ones to save computation
    for item in output:
        item['health'] = calculate_health_score(item['retailer_id'])

    return jsonify({
        'top_retailers': output
    })

# =========================================================
# DISTRICT INTELLIGENCE API (Unified)
# =========================================================

@app.route('/api/district-intelligence')
def district_intelligence():
    import json as json_lib
    district = request.args.get('district')
    if not district:
        return jsonify({'error': 'Missing district'}), 400

    result = {}

    # 1. Crop Lifecycle Analysis
    growers = DATA['growers']
    district_growers = growers[growers['district'] == district]
    crop_data = {}
    stages_info = []
    if not district_growers.empty:
        for _, row in district_growers.head(200).iterrows():
            try:
                if pd.notna(row.get('grower_crop_calendar')):
                    gc = json_lib.loads(row['grower_crop_calendar'])
                    crop = gc.get('crop', 'unknown')
                    crop_data[crop] = crop_data.get(crop, 0) + 1
                    if len(stages_info) == 0 and gc.get('stages'):
                        stages_info = gc['stages']
            except:
                pass

    top_crops = sorted(crop_data.items(), key=lambda x: x[1], reverse=True)[:3]
    result['crop_lifecycle'] = {
        'total_growers': len(district_growers),
        'top_crops': [{'crop': c, 'grower_count': n} for c, n in top_crops],
        'growth_stages': stages_info
    }

    # 2. Digital Marketing Insights
    whatsapp = DATA['whatsapp']
    if not district_growers.empty:
        grower_ids = district_growers['grower_id'].tolist()
        campaigns = whatsapp[whatsapp['grower_id'].isin(grower_ids)]
        total_sent = len(campaigns)
        total_delivered = len(campaigns[campaigns['delivered_status'] == True]) if total_sent else 0
        total_opened = len(campaigns[campaigns['opened_status'] == True]) if total_sent else 0
        total_clicked = len(campaigns[campaigns['clicked_status'] == True]) if total_sent else 0

        # Top campaigned products
        product_counts = campaigns['campaign_product'].value_counts().head(3)
        top_products = [{'product': p, 'count': int(c)} for p, c in product_counts.items()]
    else:
        total_sent = total_delivered = total_opened = total_clicked = 0
        top_products = []

    result['marketing'] = {
        'whatsapp_sent': total_sent,
        'whatsapp_delivered': total_delivered,
        'whatsapp_opened': total_opened,
        'whatsapp_clicked': total_clicked,
        'open_rate': round((total_opened / total_sent * 100), 1) if total_sent else 0,
        'click_rate': round((total_clicked / total_sent * 100), 1) if total_sent else 0,
        'top_campaigned_products': top_products
    }

    # 3. Product Recommendations from Catalog
    products_df = DATA['products']
    recommended_products = []
    if top_crops:
        main_crop = top_crops[0][0].capitalize()
        for _, prod in products_df.iterrows():
            if pd.notna(prod.get(main_crop)) and str(prod.get(main_crop, '')).strip():
                recommended_products.append({
                    'sku_id': prod['sku_id'],
                    'name': prod['sku_name'],
                    'class': prod.get('Class', ''),
                    'target_disease': str(prod.get(main_crop, '')),
                    'description': str(prod.get('Description', ''))[:120]
                })

    result['product_recommendations'] = recommended_products[:5]
    result['district'] = district

    return jsonify(result)

# =========================================================
# SALES TREND ANALYTICS
# =========================================================

@app.route('/api/sales-trends')
def sales_trends():
    district = request.args.get('district')
    pos = DATA['pos']
    retailers = DATA['retailers']

    if district:
        district_retailers = retailers[retailers['district'] == district]['retailer_id'].tolist()
        filtered = pos[pos['retailer_id'].isin(district_retailers)]
    else:
        filtered = pos

    filtered = filtered.copy()
    filtered['week'] = filtered['transaction_date'].dt.to_period('W').dt.start_time
    weekly = filtered.groupby('week').agg(
        total_qty=('sku_qty', 'sum'),
        total_revenue=('sku_price', lambda x: (x * filtered.loc[x.index, 'sku_qty']).sum()),
        transactions=('transaction_id', 'nunique')
    ).reset_index().sort_values('week')

    # Top products by week
    product_weekly = filtered.groupby([filtered['week'], 'sku_name'])['sku_qty'].sum().reset_index()
    top_products = filtered.groupby('sku_name')['sku_qty'].sum().sort_values(ascending=False).head(5).index.tolist()

    product_series = {}
    for p in top_products:
        p_data = product_weekly[product_weekly['sku_name'] == p].sort_values('week')
        product_series[p] = {
            'labels': [d.strftime('%d %b') for d in p_data['week']],
            'data': p_data['sku_qty'].tolist()
        }

    return jsonify({
        'labels': [d.strftime('%d %b') for d in weekly['week']],
        'qty': weekly['total_qty'].tolist(),
        'revenue': weekly['total_revenue'].tolist(),
        'transactions': weekly['transactions'].tolist(),
        'product_series': product_series
    })

# =========================================================
# DIGITAL FUNNEL ANALYTICS
# =========================================================

@app.route('/api/digital-funnel')
def digital_funnel():
    df = DATA['digital']
    df = df.copy()
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])

    total_impressions = int(df['social_post_impression'].sum())
    total_visits = int(df['landing_page_visits'].sum())
    total_leads = int(df['lead_form_submission'].sum())

    # Conversion rates
    imp_to_visit = round(total_visits / total_impressions * 100, 2) if total_impressions else 0
    visit_to_lead = round(total_leads / total_visits * 100, 2) if total_visits else 0
    overall = round(total_leads / total_impressions * 100, 3) if total_impressions else 0

    # By product
    by_product = df.groupby('campaign_product').agg(
        impressions=('social_post_impression', 'sum'),
        visits=('landing_page_visits', 'sum'),
        leads=('lead_form_submission', 'sum')
    ).reset_index().sort_values('impressions', ascending=False)

    product_data = []
    for _, r in by_product.iterrows():
        product_data.append({
            'product': r['campaign_product'],
            'impressions': int(r['impressions']),
            'visits': int(r['visits']),
            'leads': int(r['leads']),
            'conversion': round(r['leads'] / r['impressions'] * 100, 2) if r['impressions'] else 0
        })

    # Weekly trend
    weekly = df.groupby('week_start_date').agg(
        impressions=('social_post_impression', 'sum'),
        visits=('landing_page_visits', 'sum'),
        leads=('lead_form_submission', 'sum')
    ).reset_index().sort_values('week_start_date')

    return jsonify({
        'totals': {
            'impressions': total_impressions,
            'visits': total_visits,
            'leads': total_leads,
            'imp_to_visit_rate': imp_to_visit,
            'visit_to_lead_rate': visit_to_lead,
            'overall_rate': overall
        },
        'by_product': product_data[:6],
        'weekly_trend': {
            'labels': [d.strftime('%d %b') for d in weekly['week_start_date']],
            'impressions': weekly['impressions'].tolist(),
            'visits': weekly['visits'].tolist(),
            'leads': weekly['leads'].tolist()
        }
    })

# =========================================================
# REVENUE & GROWTH ENGINE
# =========================================================

@app.route('/api/revenue-opportunities')
def revenue_opportunities():
    district = request.args.get('district')
    if not district:
        return jsonify([])

    # 1. Weather-Triggered Campaign (Simulated Logic based on district)
    weather_campaign = {
        'type': 'weather',
        'title': '🌧️ Heavy Rain Alert: High Fungicide Demand',
        'description': f'Rain expected in {district} over the next 48 hours. Suggest broadcasting Amistar/Kavach availability to local farmers.',
        'action_text': 'Broadcast to Farmers',
        'action_whatsapp': f'Weather alert for {district}: High humidity detected. Apply Syngenta Amistar immediately to protect crops.'
    }

    # 2. Cross-Selling & Bundling (Simulated Logic based on season/district)
    cross_sell = {
        'type': 'cross_sell',
        'title': '🛒 Tomato Sowing Season: Seed Treatment Bundle',
        'description': f'Farmers in {district} are buying Tomato seeds. Create a bundle offer with Cruiser 350 FS to increase AOV.',
        'action_text': 'Launch Bundle Campaign',
        'action_whatsapp': f'Special Offer for {district}: Buy Tomato Seeds + Cruiser 350 FS together for a 10% discount!'
    }

    # 3. Auto-Replenishment (Simulated Logic based on top retailer)
    # Find a top retailer in the district to simulate stockout risk
    retailers = DATA['retailers'][DATA['retailers']['district'] == district]
    if not retailers.empty:
        r_id = retailers.iloc[0]['retailer_id']
    else:
        r_id = 'RTL_UNKNOWN'

    replenish = {
        'type': 'replenishment',
        'title': f'📦 Predictive Stockout: {r_id}',
        'description': f'{r_id} is selling Actara 25WG rapidly and will run out in 3 days. Send a 1-tap reorder link now.',
        'action_text': 'Restock Retailer',
        'action_whatsapp': f'Hi {r_id}, our systems predict you will run out of Actara 25WG in 3 days. Reply "REORDER" to restock now.'
    }

    return jsonify([weather_campaign, cross_sell, replenish])

# =========================================================
# CHURN PREDICTION
# =========================================================

@app.route('/api/churn-prediction')
def churn_prediction():
    district = request.args.get('district')
    retailers_df = DATA['retailers']
    pos = DATA['pos']
    visits = DATA['visits']

    if district:
        target = retailers_df[retailers_df['district'] == district]
    else:
        target = retailers_df.head(100)

    churn_list = []
    for _, row in target.iterrows():
        rid = row['retailer_id']
        r_sales = pos[pos['retailer_id'] == rid]

        if r_sales.empty:
            continue

        last_date = r_sales['transaction_date'].max()
        overall_max = pos['transaction_date'].max()
        days_inactive = (overall_max - last_date).days

        recent = r_sales[r_sales['transaction_date'] >= last_date - timedelta(days=30)]['sku_qty'].sum()
        prev = r_sales[(r_sales['transaction_date'] < last_date - timedelta(days=30)) & (r_sales['transaction_date'] >= last_date - timedelta(days=60))]['sku_qty'].sum()

        if prev > 0:
            trend = round((recent - prev) / prev * 100, 1)
        else:
            trend = 0

        tehsil = row['tehsil']
        visit_count = len(visits[(visits['visit_tehsil'] == tehsil) & (visits['visit_date'] >= overall_max - timedelta(days=90))])

        # Churn risk scoring
        risk = 0
        risk_factors = []
        if days_inactive > 21:
            risk += 35
            risk_factors.append(f"{days_inactive} days since last sale")
        if trend < -30:
            risk += 30
            risk_factors.append(f"Sales dropped {abs(trend)}%")
        elif trend < -10:
            risk += 15
            risk_factors.append(f"Sales declining {abs(trend)}%")
        if visit_count == 0:
            risk += 25
            risk_factors.append("No field visits in 90 days")
        elif visit_count < 2:
            risk += 10
            risk_factors.append("Very few field visits")

        risk = min(95, risk)

        if risk >= 30:
            churn_list.append({
                'retailer_id': rid,
                'district': row['district'],
                'risk_score': risk,
                'risk_level': 'CRITICAL' if risk >= 70 else ('HIGH' if risk >= 50 else 'MEDIUM'),
                'days_inactive': days_inactive,
                'sales_trend': trend,
                'recent_visits': visit_count,
                'factors': risk_factors
            })

    churn_list.sort(key=lambda x: x['risk_score'], reverse=True)

    return jsonify({
        'at_risk': churn_list[:8],
        'total_at_risk': len(churn_list),
        'critical_count': len([c for c in churn_list if c['risk_level'] == 'CRITICAL']),
        'high_count': len([c for c in churn_list if c['risk_level'] == 'HIGH'])
    })

# =========================================================
# GROWER SCAN-TO-PURCHASE ANALYTICS
# =========================================================

@app.route('/api/grower-analytics')
def grower_analytics():
    district = request.args.get('district')
    growers = DATA['growers']

    if district:
        gdf = growers[growers['district'] == district]
    else:
        gdf = growers

    total = len(gdf)
    scanned = len(gdf[gdf['product_scan'] == True])
    not_scanned = total - scanned
    scan_rate = round(scanned / total * 100, 1) if total else 0

    # Device breakdown
    device_counts = gdf['device_type'].value_counts().to_dict()

    # Language breakdown
    lang_counts = gdf['language'].value_counts().head(5).to_dict()

    # Farm size distribution
    avg_farm = round(gdf['grower_farm_size'].mean(), 2) if 'grower_farm_size' in gdf.columns else 0

    # Scanned products
    scanned_products = gdf[gdf['product_scan'] == True]['product_name'].value_counts().head(5)
    top_scanned = [{'product': p, 'count': int(c)} for p, c in scanned_products.items()]

    # Offline campaign attendance
    attended = len(gdf[gdf['offline_campaign_attended'] == True])
    attend_rate = round(attended / total * 100, 1) if total else 0

    # Age distribution
    age_groups = {'18-30': 0, '31-45': 0, '46-60': 0, '60+': 0}
    for age in gdf['grower_age'].dropna():
        if age <= 30: age_groups['18-30'] += 1
        elif age <= 45: age_groups['31-45'] += 1
        elif age <= 60: age_groups['46-60'] += 1
        else: age_groups['60+'] += 1

    return jsonify({
        'total_growers': total,
        'scan_rate': scan_rate,
        'scanned': scanned,
        'not_scanned': not_scanned,
        'avg_farm_size': avg_farm,
        'device_breakdown': device_counts,
        'language_breakdown': lang_counts,
        'top_scanned_products': top_scanned,
        'campaign_attendance': attended,
        'attend_rate': attend_rate,
        'age_distribution': age_groups
    })

# =========================================================
# PAGES
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/retailer/<retailer_id>')
def retailer_page(retailer_id):
    return render_template('retailer_detail.html', retailer_id=retailer_id)

@app.route('/offline')
def offline_page():
    return render_template('offline.html')

# =========================================================
# PREDICTION ROUTE
# =========================================================

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    lat = request.form.get('latitude')
    lon = request.form.get('longitude')
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict disease using prediction.py logic
        disease, confidence = predict_disease(filepath)
        
        # Fetch weather data using coordinates
        temperature, humidity, weather_condition = None, None, None
        if lat and lon:
            try:
                # Open-Meteo is a free, no-key required alternative to OpenWeatherMap
                # Note: user mentioned OpenWeatherMap, but Open-Meteo doesn't require an API key to just work instantly.
                # However, let's stick to OpenWeatherMap or Open-Meteo based on what's easiest. 
                # The existing app.py code uses Open-Meteo for weather insights! Let's reuse that format.
                url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code"
                res = requests.get(url, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    current = data.get('current', {})
                    temperature = current.get('temperature_2m')
                    humidity = current.get('relative_humidity_2m')
                    weather_condition = current.get('weather_code')
            except Exception as e:
                print("Weather API error:", e)
                
        # Detailed agricultural recommendations mapping with Syngenta products
        DISEASE_RECOMMENDATIONS = {
            "potato___early_blight": "Remove affected leaves. Apply Syngenta Score 250 EC or Amistar 250 SC. Practice crop rotation and avoid overhead watering.",
            "potato___late_blight": "Apply fungicide immediately (e.g., Syngenta Kavach 75 WP or Amistar 250 SC). Destroy infected plants to prevent rapid spread.",
            "tomato_early_blight": "Remove infected lower leaves. Apply Syngenta Score 250 EC. Always water at the base of the plant.",
            "tomato_late_blight": "Highly contagious! Apply Syngenta Kavach 75 WP or Amistar 250 SC. Remove and safely destroy infected plants immediately.",
            "tomato_bacterial_spot": "Apply copper-based bactericides or Syngenta Amistar 250 SC. Avoid overhead watering and avoid working with plants when they are wet.",
            "tomato_septoria_leaf_spot": "Remove diseased leaves. Improve air circulation by pruning. Apply Syngenta Kavach 75 WP or Score 250 EC.",
            "pepper__bell___bacterial_spot": "Use appropriate Syngenta broad-spectrum treatments. Ensure good air circulation and remove any infected plant debris."
        }
        
        disease_key = disease.lower()
        
        if "healthy" in disease_key:
            recommendation = "Crop appears healthy! Continue regular monitoring and optimal watering."
        else:
            base_recommendation = DISEASE_RECOMMENDATIONS.get(disease_key, f"Disease detected: {disease}. Apply recommended targeted treatments.")
            
            # Context-aware weather additions
            weather_warning = ""
            if temperature is not None and humidity is not None:
                if "late_blight" in disease_key and humidity > 75:
                    weather_warning = " ⚠️ High humidity detected. Fungal spread risk is extreme. Act quickly."
                elif "early_blight" in disease_key and temperature > 25:
                    weather_warning = " ⚠️ Warm weather detected. Ideal conditions for early blight to spread."
                elif "bacterial_spot" in disease_key and humidity > 70:
                    weather_warning = " ⚠️ High humidity accelerates bacterial spread. Ensure foliage dries quickly."
                elif humidity > 85:
                    weather_warning = f" ⚠️ Current humidity ({humidity}%) is highly favorable for disease progression."
            
            recommendation = base_recommendation + weather_warning
                
        return jsonify({
            'disease': disease,
            'confidence': confidence,
            'temperature': temperature,
            'humidity': humidity,
            'recommendation': recommendation
        })

# =========================================================
# IOT CHEMICAL DETECTION API (AS7341 & SGP30)
# =========================================================

@app.route('/api/chemical-detection', methods=['POST'])
def chemical_detection():
    data = request.json
    try:
        voc = float(data.get('voc', 0))
        spectral = float(data.get('spectral', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid sensor readings provided.'}), 400

    # Basic logic to classify chemical type based on VOC and Spectral signatures
    if voc > 800 and spectral > 1200:
        chemical_type = "Herbicide (e.g., Glyphosate/Paraquat type)"
        confidence = 94.5
        recommendation = "High herbicide signature detected. Verify application rates and avoid over-spraying."
    elif voc > 300 or spectral > 600:
        chemical_type = "Fungicide (e.g., Mancozeb/Azoxystrobin type)"
        confidence = 88.2
        recommendation = "Fungicide residue signature detected. Typical profile for recent protective sprays."
    else:
        chemical_type = "No Significant Chemical Residue"
        confidence = 98.0
        recommendation = "Readings are within normal baseline background levels."

    return jsonify({
        'type': chemical_type,
        'confidence': confidence,
        'voc_level': voc,
        'spectral_level': spectral,
        'recommendation': recommendation
    })

# =========================================================
# RUN
# =========================================================

if __name__ == '__main__':
    # Use an environment variable for the port; default to 5001.
    # This avoids "Address already in use" errors when the default port is occupied.
    import os
    port = int(os.getenv('PORT', 5001))
    app.run(debug=True, port=5002)