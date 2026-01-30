#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
╔═══════════════════════════════════════════════════════════╗
║                      SENO AI BOT                          ║
║              بوت ذكاء اصطناعي متطور                      ║
║        Multi-AI Support (4 Free APIs!)                    ║
║                  جميع الحقوق محفوظة                      ║
╚═══════════════════════════════════════════════════════════╝
"""

import os
import sqlite3
import asyncio
import re
from datetime import datetime
from typing import Optional, Dict, List
import google.generativeai as genai
from groq import Groq
import anthropic
from telebot.async_telebot import AsyncTeleBot
from telebot import types
import logging
import random

# ═══════════════════════════════════════════════════════════
# إعداد السجلات
# ═══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# الإعدادات الأساسية
# ═══════════════════════════════════════════════════════════
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN')

# مفاتيح الـ APIs المجانية (اختياري - ضع ما لديك)
# يمكنك إضافة عدة مفاتيح Gemini (افصل بينها بفاصلة)
GEMINI_API_KEYS = os.getenv('GEMINI_API_KEYS', '').split(',') if os.getenv('GEMINI_API_KEYS') else []
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', '')

ADMIN_IDS = [int(x.strip()) for x in os.getenv('ADMIN_IDS', '123456789').split(',') if x.strip()]
CHANNEL_USERNAME = os.getenv('CHANNEL_USERNAME', '@your_channel')
CHANNEL_ID = int(os.getenv('CHANNEL_ID', '-1001234567890'))

bot = AsyncTeleBot(TELEGRAM_TOKEN, parse_mode='HTML')

# متغيرات عامة
broadcast_mode = {}
waiting_for_user_id = {}

# ═══════════════════════════════════════════════════════════
# نظام الذكاء الاصطناعي المتعدد
# ═══════════════════════════════════════════════════════════
class MultiAI:
    def __init__(self):
        self.apis = []
        self.current_api_index = 0
        
        # تهيئة Google Gemini APIs (دعم مفاتيح متعددة!)
        if GEMINI_API_KEYS:
            for idx, api_key in enumerate(GEMINI_API_KEYS):
                api_key = api_key.strip()
                if not api_key:
                    continue
                try:
                    genai.configure(api_key=api_key)
                    generation_config = {
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 8192,
                    }
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                    gemini_model = genai.GenerativeModel(
                        model_name='gemini-pro',
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    self.apis.append({
                        'name': f'Gemini #{idx + 1}',
                        'client': gemini_model,
                        'type': 'gemini',
                        'icon': '💎',
                        'api_key': api_key
                    })
                    logger.info(f"✅ تم تفعيل Google Gemini API #{idx + 1}")
                except Exception as e:
                    logger.error(f"خطأ في تهيئة Gemini #{idx + 1}: {e}")
        
        # تهيئة Groq (مجاني - سريع جداً!)
        if GROQ_API_KEY:
            try:
                groq_client = Groq(api_key=GROQ_API_KEY)
                self.apis.append({
                    'name': 'Groq',
                    'client': groq_client,
                    'type': 'groq',
                    'icon': '⚡',
                    'model': 'llama-3.3-70b-versatile'
                })
                logger.info("✅ تم تفعيل Groq API")
            except Exception as e:
                logger.error(f"خطأ في تهيئة Groq: {e}")
        
        # تهيئة HuggingFace (مجاني)
        if HUGGINGFACE_API_KEY:
            self.apis.append({
                'name': 'HuggingFace',
                'api_key': HUGGINGFACE_API_KEY,
                'type': 'huggingface',
                'icon': '🤗',
                'model': 'meta-llama/Llama-3.2-3B-Instruct'
            })
            logger.info("✅ تم تفعيل HuggingFace API")
        
        # تهيئة Together AI (مجاني - $25 رصيد بداية)
        if TOGETHER_API_KEY:
            self.apis.append({
                'name': 'Together',
                'api_key': TOGETHER_API_KEY,
                'type': 'together',
                'icon': '🌟',
                'model': 'meta-llama/Llama-3-70b-chat-hf'
            })
            logger.info("✅ تم تفعيل Together AI API")
        
        if not self.apis:
            logger.warning("⚠️ لم يتم تفعيل أي API! الرجاء إضافة مفتاح واحد على الأقل")
        else:
            logger.info(f"🚀 تم تفعيل {len(self.apis)} APIs")
    
    def get_system_prompt(self, user_name: str) -> str:
        """النص التوجيهي للنظام"""
        return f"""أنت Seno AI، مساعد ذكاء اصطناعي متطور وذكي جداً.

مهامك:
- الرد على جميع الأسئلة بشكل احترافي ومفيد
- كتابة الأكواد البرمجية بجميع اللغات بشكل احترافي
- المساعدة في حل المشاكل التقنية والبرمجية
- تقديم معلومات دقيقة وموثوقة
- الرد باللغة العربية بشكل أساسي، وبأي لغة يطلبها المستخدم

عند كتابة الأكواد:
- استخدم تنسيق مرتب ومنظم جداً
- أضف تعليقات توضيحية بالعربية
- اجعل الكود قابلاً للنسخ بسهولة
- ضع الكود داخل كتل برمجية بهذا الشكل:
```python
# كود هنا
```

الأسلوب:
- كن ودوداً واحترافياً جداً
- أجب بشكل مفصل وواضح
- استخدم الرموز التعبيرية بشكل مناسب
- نظم إجابتك بشكل جميل ومرتب

اسم المستخدم الحالي: {user_name}"""
    
    async def _call_gemini(self, api_info: Dict, user_message: str, user_name: str) -> str:
        """استدعاء Gemini API"""
        try:
            # إعداد المفتاح الخاص بهذا الـ API
            genai.configure(api_key=api_info['api_key'])
            
            prompt = f"{self.get_system_prompt(user_name)}\n\nالسؤال: {user_message}"
            response = api_info['client'].generate_content(prompt)
            if response.text:
                return response.text
            return None
        except Exception as e:
            logger.error(f"خطأ في {api_info['name']}: {e}")
            return None
    
    async def _call_groq(self, api_info: Dict, user_message: str, user_name: str) -> str:
        """استدعاء Groq API"""
        try:
            chat_completion = api_info['client'].chat.completions.create(
                messages=[
                    {"role": "system", "content": self.get_system_prompt(user_name)},
                    {"role": "user", "content": user_message}
                ],
                model=api_info['model'],
                temperature=0.7,
                max_tokens=8192,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"خطأ في Groq: {e}")
            return None
    
    async def _call_huggingface(self, api_info: Dict, user_message: str, user_name: str) -> str:
        """استدعاء HuggingFace API"""
        try:
            import requests
            
            API_URL = f"https://api-inference.huggingface.co/models/{api_info['model']}"
            headers = {"Authorization": f"Bearer {api_info['api_key']}"}
            
            prompt = f"{self.get_system_prompt(user_name)}\n\nالسؤال: {user_message}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.95,
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').replace(prompt, '').strip()
            return None
        except Exception as e:
            logger.error(f"خطأ في HuggingFace: {e}")
            return None
    
    async def _call_together(self, api_info: Dict, user_message: str, user_name: str) -> str:
        """استدعاء Together AI API"""
        try:
            import requests
            
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_info['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": api_info['model'],
                "messages": [
                    {"role": "system", "content": self.get_system_prompt(user_name)},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 4096,
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            return None
        except Exception as e:
            logger.error(f"خطأ في Together AI: {e}")
            return None
    
    async def get_response(self, user_message: str, user_name: str = "المستخدم") -> tuple:
        """
        الحصول على رد من أحد الـ APIs
        يحاول جميع الـ APIs تلقائياً حتى يحصل على رد
        Returns: (response_text, api_name, api_icon)
        """
        if not self.apis:
            return ("❌ عذراً، لم يتم تفعيل أي API! الرجاء التواصل مع المطور.", "None", "❌")
        
        # نسخة من قائمة الـ APIs للمحاولة
        apis_to_try = self.apis.copy()
        
        # خلط القائمة للتوزيع العادل
        random.shuffle(apis_to_try)
        
        for api_info in apis_to_try:
            try:
                logger.info(f"🔄 محاولة استخدام {api_info['name']} API...")
                
                response = None
                
                if api_info['type'] == 'gemini':
                    response = await self._call_gemini(api_info, user_message, user_name)
                elif api_info['type'] == 'groq':
                    response = await self._call_groq(api_info, user_message, user_name)
                elif api_info['type'] == 'huggingface':
                    response = await self._call_huggingface(api_info, user_message, user_name)
                elif api_info['type'] == 'together':
                    response = await self._call_together(api_info, user_message, user_name)
                
                if response and len(response.strip()) > 0:
                    logger.info(f"✅ تم الحصول على رد من {api_info['name']}")
                    return (response, api_info['name'], api_info['icon'])
                else:
                    logger.warning(f"⚠️ رد فارغ من {api_info['name']}, جاري التحويل...")
                    
            except Exception as e:
                logger.error(f"❌ خطأ في {api_info['name']}: {e}")
                continue
        
        # إذا فشلت جميع الـ APIs
        return ("❌ عذراً، جميع خدمات الذكاء الاصطناعي غير متاحة حالياً. الرجاء المحاولة بعد قليل.", "Failed", "❌")

ai = MultiAI()

# ═══════════════════════════════════════════════════════════
# قاعدة البيانات
# ═══════════════════════════════════════════════════════════
class Database:
    def __init__(self, db_name='seno_ai_bot.db'):
        self.db_name = db_name
        self.init_db()
    
    def get_connection(self):
        return sqlite3.connect(self.db_name, check_same_thread=False)
    
    def init_db(self):
        """تهيئة قاعدة البيانات"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                join_date TEXT,
                message_count INTEGER DEFAULT 0,
                is_blocked INTEGER DEFAULT 0,
                last_active TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                user_message TEXT,
                bot_response TEXT,
                ai_used TEXT,
                timestamp TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS broadcasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT,
                sent_count INTEGER DEFAULT 0,
                failed_count INTEGER DEFAULT 0,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ تم تهيئة قاعدة البيانات بنجاح")
    
    def add_user(self, user_id: int, username: str = None, first_name: str = None, last_name: str = None):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, username, first_name, last_name, join_date, last_active)
                VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT join_date FROM users WHERE user_id = ?), ?),
                    ?)
            ''', (user_id, username, first_name, last_name, user_id, now, now))
            conn.commit()
        except Exception as e:
            logger.error(f"خطأ في إضافة المستخدم: {e}")
        finally:
            conn.close()
    
    def update_user_activity(self, user_id: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('UPDATE users SET message_count = message_count + 1, last_active = ? WHERE user_id = ?', (now, user_id))
        conn.commit()
        conn.close()
    
    def save_conversation(self, user_id: int, user_message: str, bot_response: str, ai_used: str = 'Unknown'):
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO conversations (user_id, user_message, bot_response, ai_used, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, user_message, bot_response, ai_used, timestamp))
        conn.commit()
        conn.close()
    
    def get_user_info(self, user_id: int) -> Optional[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                'user_id': row[0],
                'username': row[1],
                'first_name': row[2],
                'last_name': row[3],
                'join_date': row[4],
                'message_count': row[5],
                'is_blocked': row[6],
                'last_active': row[7]
            }
        return None
    
    def get_statistics(self) -> Dict:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_messages = cursor.fetchone()[0]
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM conversations WHERE DATE(timestamp) = ?', (today,))
        active_today = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_blocked = 1')
        blocked_users = cursor.fetchone()[0]
        conn.close()
        return {
            'total_users': total_users,
            'total_messages': total_messages,
            'active_today': active_today,
            'blocked_users': blocked_users
        }
    
    def get_all_users(self) -> List[int]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT user_id FROM users WHERE is_blocked = 0')
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        return users
    
    def block_user(self, user_id: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_blocked = 1 WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
    
    def unblock_user(self, user_id: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_blocked = 0 WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
    
    def is_user_blocked(self, user_id: int) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT is_blocked FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] == 1 if result else False
    
    def save_broadcast(self, message: str, sent: int, failed: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('INSERT INTO broadcasts (message, sent_count, failed_count, timestamp) VALUES (?, ?, ?, ?)', (message, sent, failed, timestamp))
        conn.commit()
        conn.close()

db = Database()

# ═══════════════════════════════════════════════════════════
# معالج تنسيق الرسائل
# ═══════════════════════════════════════════════════════════
def format_code_response(text: str) -> str:
    """تنسيق الردود لتحسين عرض الأكواد"""
    code_pattern = r'```(\w+)?\n(.*?)```'
    def replace_code(match):
        language = match.group(1) or 'Code'
        code = match.group(2)
        formatted = f"""
<b>📝 {language.capitalize()}</b>
<pre><code class="language-{language.lower()}">{code}</code></pre>"""
        return formatted
    formatted_text = re.sub(code_pattern, replace_code, text, flags=re.DOTALL)
    return formatted_text

# ═══════════════════════════════════════════════════════════
# التحقق من الاشتراك الإجباري
# ═══════════════════════════════════════════════════════════
async def check_subscription(user_id: int) -> bool:
    try:
        if user_id in ADMIN_IDS:
            return True
        member = await bot.get_chat_member(CHANNEL_ID, user_id)
        return member.status in ['member', 'administrator', 'creator']
    except Exception as e:
        logger.error(f"خطأ في التحقق من الاشتراك: {e}")
        return False

async def send_subscription_message(chat_id: int, user_name: str):
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(
        types.InlineKeyboardButton("📢 الاشتراك في القناة", url=f"https://t.me/{CHANNEL_USERNAME.replace('@', '')}"),
        types.InlineKeyboardButton("✅ تحقق من الاشتراك", callback_data="check_subscription")
    )
    text = f"""
🔒 <b>الاشتراك الإجباري</b>

عزيزي <b>{user_name}</b> 👋

للاستفادة من خدمات <b>Seno AI</b> المتطورة، يجب عليك الاشتراك في قناتنا الرسمية أولاً.

<b>⬇️ خطوات بسيطة:</b>
1️⃣ اضغط على زر "الاشتراك في القناة"
2️⃣ اشترك في القناة
3️⃣ ارجع واضغط على "تحقق من الاشتراك"
4️⃣ ابدأ باستخدام البوت مجاناً! 🚀

💎 <b>قناتنا:</b> {CHANNEL_USERNAME}
"""
    await bot.send_message(chat_id, text, reply_markup=keyboard)

# ═══════════════════════════════════════════════════════════
# لوحات المفاتيح
# ═══════════════════════════════════════════════════════════
def get_main_keyboard(is_admin: bool = False):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [
        types.KeyboardButton("💬 محادثة جديدة"),
        types.KeyboardButton("📊 إحصائياتي"),
        types.KeyboardButton("ℹ️ معلومات البوت"),
        types.KeyboardButton("📞 المطور")
    ]
    if is_admin:
        buttons.append(types.KeyboardButton("👨‍💼 لوحة التحكم"))
    keyboard.add(*buttons)
    return keyboard

def get_admin_keyboard():
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [
        types.KeyboardButton("📊 الإحصائيات الكاملة"),
        types.KeyboardButton("📢 إذاعة رسالة"),
        types.KeyboardButton("👥 عدد المستخدمين"),
        types.KeyboardButton("🔍 بحث عن مستخدم"),
        types.KeyboardButton("🚫 حظر مستخدم"),
        types.KeyboardButton("✅ إلغاء الحظر"),
        types.KeyboardButton("🔙 القائمة الرئيسية")
    ]
    keyboard.add(*buttons)
    return keyboard

# ═══════════════════════════════════════════════════════════
# معالجات الأوامر
# ═══════════════════════════════════════════════════════════
@bot.message_handler(commands=['start'])
async def start_command(message):
    user_id = message.from_user.id
    username = message.from_user.username
    first_name = message.from_user.first_name
    last_name = message.from_user.last_name
    
    db.add_user(user_id, username, first_name, last_name)
    
    if db.is_user_blocked(user_id):
        await bot.send_message(message.chat.id, "⛔️ <b>عذراً!</b>\n\nتم حظرك من استخدام هذا البوت.\n\nللاستفسار تواصل مع المطور.")
        return
    
    is_subscribed = await check_subscription(user_id)
    if not is_subscribed:
        await send_subscription_message(message.chat.id, first_name)
        return
    
    # عرض الـ APIs المفعلة
    apis_text = "\n".join([f"{api['icon']} {api['name']}" for api in ai.apis]) if ai.apis else "❌ لا يوجد"
    
    welcome_text = f"""
╔═══════════════════════════════════╗
║   <b>مرحباً بك في Seno AI</b> 🤖   ║
╚═══════════════════════════════════╝

أهلاً وسهلاً <b>{first_name}</b>! 👋

أنا <b>Seno AI</b>، مساعدك الذكي المتطور! ✨

<b>🎯 ماذا أستطيع أن أفعل؟</b>

💻 <b>البرمجة والأكواد</b>
• كتابة أكواد احترافية بجميع اللغات
• إصلاح الأخطاء البرمجية
• شرح الأكواد المعقدة

🧠 <b>الذكاء والمعرفة</b>
• الإجابة على جميع أسئلتك
• المساعدة في حل المشاكل
• تقديم معلومات دقيقة

✍️ <b>الكتابة والإبداع</b>
• كتابة المقالات والمحتوى
• تحسين النصوص
• الأفكار الإبداعية

<b>🤖 الـ AI المفعلة ({len(ai.apis)}):</b>
{apis_text}

<b>🚀 ابدأ الآن!</b>
فقط أرسل رسالتك أو سؤالك وسأساعدك فوراً!

━━━━━━━━━━━━━━━━━━━━━
<i>Multi-AI System 🌟</i>
<i>100% مجاني! 🎉</i>
"""
    
    is_admin = user_id in ADMIN_IDS
    await bot.send_message(message.chat.id, welcome_text, reply_markup=get_main_keyboard(is_admin))

@bot.message_handler(commands=['help'])
async def help_command(message):
    help_text = """
<b>📖 دليل الاستخدام</b>

<b>الأوامر المتاحة:</b>
/start - بدء البوت
/help - عرض المساعدة
/stats - إحصائياتك الشخصية
/cancel - إلغاء العملية الحالية

<b>🤖 كيف تستخدم البوت؟</b>
فقط أرسل أي رسالة أو سؤال وسأجيبك فوراً!

<b>💡 أمثلة:</b>
• "اكتب لي كود Python لحساب الأعداد الأولية"
• "ما هي أفضل طريقة لتعلم البرمجة؟"
• "ساعدني في حل هذه المشكلة..."

<b>✨ مميزات خاصة:</b>
• نظام AI متعدد - يحول تلقائياً!
• ردود فورية وذكية
• كتابة أكواد احترافية
• <b>100% مجاني!</b> 🎉
"""
    await bot.send_message(message.chat.id, help_text)

@bot.message_handler(commands=['stats'])
async def stats_command(message):
    user_id = message.from_user.id
    user_info = db.get_user_info(user_id)
    if user_info:
        stats_text = f"""
<b>📊 إحصائياتك الشخصية</b>

👤 <b>الاسم:</b> {user_info['first_name']}
🆔 <b>المعرف:</b> <code>{user_info['user_id']}</code>
📅 <b>تاريخ الانضمام:</b> {user_info['join_date'][:10]}
💬 <b>عدد الرسائل:</b> {user_info['message_count']}
🕐 <b>آخر نشاط:</b> {user_info['last_active'][:16]}

شكراً لاستخدامك <b>Seno AI</b>! 🌟
"""
        await bot.send_message(message.chat.id, stats_text)
    else:
        await bot.send_message(message.chat.id, "❌ لم يتم العثور على معلوماتك")

@bot.message_handler(func=lambda message: message.text == "💬 محادثة جديدة")
async def new_chat(message):
    await bot.send_message(message.chat.id, "✨ <b>محادثة جديدة بدأت!</b>\n\nأرسل رسالتك أو سؤالك الآن... 💭")

@bot.message_handler(func=lambda message: message.text == "📊 إحصائياتي")
async def my_stats(message):
    await stats_command(message)

@bot.message_handler(func=lambda message: message.text == "ℹ️ معلومات البوت")
async def bot_info(message):
    apis_text = "\n".join([f"{api['icon']} {api['name']}" for api in ai.apis]) if ai.apis else "❌ لا يوجد"
    info_text = f"""
<b>🤖 معلومات البوت</b>

<b>الاسم:</b> Seno AI
<b>النوع:</b> بوت ذكاء اصطناعي متطور
<b>الإصدار:</b> 3.0 (Multi-AI)
<b>اللغات المدعومة:</b> جميع اللغات

<b>🤖 الـ AI المفعلة ({len(ai.apis)}):</b>
{apis_text}

<b>🌟 المميزات:</b>
✅ نظام AI متعدد مع تحويل تلقائي
✅ كتابة أكواد احترافية
✅ ردود سريعة ودقيقة
✅ دعم شامل لجميع المجالات
✅ <b>100% مجاني تماماً!</b> 🎉

<b>📢 القناة الرسمية:</b>
{CHANNEL_USERNAME}

<b>🔧 حالة البوت:</b> يعمل بكفاءة 100% ✓
"""
    await bot.send_message(message.chat.id, info_text)

@bot.message_handler(func=lambda message: message.text == "📞 المطور")
async def contact_dev(message):
    dev_text = """
<b>📞 التواصل مع المطور</b>

للاستفسارات والدعم الفني:

<b>المطور الرئيسي:</b> Seno
<b>التواصل:</b> @Seno

<b>🌟 يمكنك طلب:</b>
• ميزات جديدة
• حل مشاكل تقنية
• استفسارات عامة
• اقتراحات للتطوير

نسعد بخدمتك! 💙
"""
    await bot.send_message(message.chat.id, dev_text)

# معالجات المطور (Admin)
@bot.message_handler(func=lambda message: message.text == "👨‍💼 لوحة التحكم")
async def admin_panel(message):
    user_id = message.from_user.id
    if user_id not in ADMIN_IDS:
        await bot.send_message(message.chat.id, "⛔️ غير مصرح لك بالوصول!")
        return
    admin_text = """
<b>👨‍💼 لوحة تحكم المطور</b>

مرحباً بك في لوحة التحكم الخاصة 🔐

اختر الإجراء المناسب من الأزرار بالأسفل:

━━━━━━━━━━━━━━━━━━━━━
<i>Seno AI Admin Panel</i>
"""
    await bot.send_message(message.chat.id, admin_text, reply_markup=get_admin_keyboard())

@bot.message_handler(func=lambda message: message.text == "📊 الإحصائيات الكاملة")
async def full_statistics(message):
    user_id = message.from_user.id
    if user_id not in ADMIN_IDS:
        return
    stats = db.get_statistics()
    apis_text = "\n".join([f"{api['icon']} {api['name']} - Active" for api in ai.apis]) if ai.apis else "❌ لا يوجد"
    stats_text = f"""
<b>📊 الإحصائيات الكاملة</b>

👥 <b>إجمالي المستخدمين:</b> {stats['total_users']}
💬 <b>إجمالي الرسائل:</b> {stats['total_messages']}
✅ <b>المستخدمون النشطون اليوم:</b> {stats['active_today']}
🚫 <b>المستخدمون المحظورون:</b> {stats['blocked_users']}

<b>🤖 حالة الـ AI ({len(ai.apis)} مفعل):</b>
{apis_text}

📅 <b>التاريخ:</b> {datetime.now().strftime('%Y-%m-%d')}
🕐 <b>الوقت:</b> {datetime.now().strftime('%H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━
<b>حالة البوت:</b> 🟢 يعمل
"""
    await bot.send_message(message.chat.id, stats_text)

@bot.message_handler(func=lambda message: message.text == "👥 عدد المستخدمين")
async def users_count(message):
    user_id = message.from_user.id
    if user_id not in ADMIN_IDS:
        return
    users = db.get_all_users()
    total = len(users)
    await bot.send_message(message.chat.id, f"<b>👥 عدد المستخدمين</b>\n\n<b>الإجمالي:</b> {total} مستخدم")

@bot.message_handler(func=lambda message: message.text == "📢 إذاعة رسالة")
async def start_broadcast(message):
    user_id = message.from_user.id
    if user_id not in ADMIN_IDS:
        return
    broadcast_mode[user_id] = True
    await bot.send_message(message.chat.id, "📢 <b>وضع الإذاعة</b>\n\nأرسل الرسالة التي تريد إذاعتها لجميع المستخدمين:\n\n<i>يمكنك إرسال نص، صورة، فيديو، أو ملف</i>\n\n<code>/cancel</code> للإلغاء")

@bot.message_handler(func=lambda message: message.text == "🔍 بحث عن مستخدم")
async def search_user_start(message):
    user_id = message.from_user.id
    if user_id not in ADMIN_IDS:
        return
    waiting_for_user_id[user_id] = 'search'
    await bot.send_message(message.chat.id, "🔍 <b>البحث عن مستخدم</b>\n\nأرسل معرف المستخدم (ID):\n\n<code>/cancel</code> للإلغاء")

@bot.message_handler(func=lambda message: message.text == "🚫 حظر مستخدم")
async def block_user_start(message):
    user_id = message.from_user.id
    if user_id not in ADMIN_IDS:
        return
    waiting_for_user_id[user_id] = 'block'
    await bot.send_message(message.chat.id, "🚫 <b>حظر مستخدم</b>\n\nأرسل معرف المستخدم (ID) للحظر:\n\n<code>/cancel</code> للإلغاء")

@bot.message_handler(func=lambda message: message.text == "✅ إلغاء الحظر")
async def unblock_user_start(message):
    user_id = message.from_user.id
    if user_id not in ADMIN_IDS:
        return
    waiting_for_user_id[user_id] = 'unblock'
    await bot.send_message(message.chat.id, "✅ <b>إلغاء الحظر</b>\n\nأرسل معرف المستخدم (ID) لإلغاء حظره:\n\n<code>/cancel</code> للإلغاء")

@bot.message_handler(func=lambda message: message.text == "🔙 القائمة الرئيسية")
async def back_to_main(message):
    user_id = message.from_user.id
    is_admin = user_id in ADMIN_IDS
    await bot.send_message(message.chat.id, "🏠 <b>القائمة الرئيسية</b>\n\nاختر ما تريد:", reply_markup=get_main_keyboard(is_admin))

@bot.message_handler(commands=['cancel'])
async def cancel_operation(message):
    user_id = message.from_user.id
    if user_id in broadcast_mode:
        del broadcast_mode[user_id]
    if user_id in waiting_for_user_id:
        del waiting_for_user_id[user_id]
    is_admin = user_id in ADMIN_IDS
    await bot.send_message(message.chat.id, "✅ تم إلغاء العملية", reply_markup=get_main_keyboard(is_admin) if not is_admin else get_admin_keyboard())

# معالج الإذاعة
async def broadcast_message(message, admin_id):
    users = db.get_all_users()
    sent = 0
    failed = 0
    status_msg = await bot.send_message(admin_id, f"📢 <b>جارِ الإذاعة...</b>\n\n👥 المستخدمون: {len(users)}\n✅ تم الإرسال: 0\n❌ فشل: 0")
    for user_id in users:
        try:
            if message.content_type == 'text':
                await bot.send_message(user_id, message.html_text, parse_mode='HTML')
            elif message.content_type == 'photo':
                await bot.send_photo(user_id, message.photo[-1].file_id, caption=message.caption)
            elif message.content_type == 'video':
                await bot.send_video(user_id, message.video.file_id, caption=message.caption)
            elif message.content_type == 'document':
                await bot.send_document(user_id, message.document.file_id, caption=message.caption)
            sent += 1
        except Exception as e:
            failed += 1
            logger.error(f"فشل إرسال رسالة للمستخدم {user_id}: {e}")
        if (sent + failed) % 10 == 0:
            try:
                await bot.edit_message_text(f"📢 <b>جارِ الإذاعة...</b>\n\n👥 المستخدمون: {len(users)}\n✅ تم الإرسال: {sent}\n❌ فشل: {failed}", admin_id, status_msg.message_id)
            except:
                pass
        await asyncio.sleep(0.05)
    db.save_broadcast(message.text or message.caption or "رسالة", sent, failed)
    await bot.edit_message_text(f"✅ <b>اكتملت الإذاعة!</b>\n\n👥 <b>إجمالي المستخدمين:</b> {len(users)}\n✅ <b>تم الإرسال:</b> {sent}\n❌ <b>فشل:</b> {failed}\n\n⏱ <b>الوقت:</b> {datetime.now().strftime('%H:%M:%S')}", admin_id, status_msg.message_id)

@bot.callback_query_handler(func=lambda call: call.data == "check_subscription")
async def check_sub_callback(call):
    user_id = call.from_user.id
    is_subscribed = await check_subscription(user_id)
    if is_subscribed:
        await bot.answer_callback_query(call.id, "✅ تم التحقق بنجاح!", show_alert=True)
        await bot.delete_message(call.message.chat.id, call.message.message_id)
        apis_text = "\n".join([f"{api['icon']} {api['name']}" for api in ai.apis]) if ai.apis else "❌ لا يوجد"
        welcome_text = f"""
╔═══════════════════════════════════╗
║   <b>مرحباً بك في Seno AI</b> 🤖   ║
╚═══════════════════════════════════╝

أهلاً وسهلاً <b>{call.from_user.first_name}</b>! 👋

شكراً لك على الاشتراك! 💙

الآن يمكنك استخدام جميع مميزات البوت!

<b>🤖 الـ AI المفعلة ({len(ai.apis)}):</b>
{apis_text}

<b>🎯 ابدأ الآن!</b>
أرسل أي رسالة أو سؤال وسأساعدك فوراً!

━━━━━━━━━━━━━━━━━━━━━
<i>Multi-AI System 🌟</i>
<i>100% مجاني! 🎉</i>
"""
        is_admin = user_id in ADMIN_IDS
        await bot.send_message(call.message.chat.id, welcome_text, reply_markup=get_main_keyboard(is_admin))
    else:
        await bot.answer_callback_query(call.id, "❌ لم تشترك بعد!\n\nالرجاء الاشتراك في القناة أولاً ثم اضغط التحقق مرة أخرى.", show_alert=True)

# معالج الرسائل الرئيسي
@bot.message_handler(content_types=['text'])
async def handle_text_message(message):
    user_id = message.from_user.id
    user_name = message.from_user.first_name
    text = message.text
    
    if db.is_user_blocked(user_id):
        await bot.send_message(message.chat.id, "⛔️ <b>عذراً!</b>\n\nتم حظرك من استخدام هذا البوت.")
        return
    
    is_subscribed = await check_subscription(user_id)
    if not is_subscribed:
        await send_subscription_message(message.chat.id, user_name)
        return
    
    if user_id in ADMIN_IDS and user_id in broadcast_mode:
        del broadcast_mode[user_id]
        await broadcast_message(message, user_id)
        return
    
    if user_id in ADMIN_IDS and user_id in waiting_for_user_id:
        action = waiting_for_user_id[user_id]
        del waiting_for_user_id[user_id]
        try:
            target_user_id = int(text)
            user_info = db.get_user_info(target_user_id)
            if action == 'search':
                if user_info:
                    info_text = f"""
<b>🔍 معلومات المستخدم</b>

👤 <b>الاسم:</b> {user_info['first_name']} {user_info['last_name'] or ''}
🆔 <b>المعرف:</b> <code>{user_info['user_id']}</code>
👨‍💼 <b>اليوزر:</b> @{user_info['username'] or 'لا يوجد'}
📅 <b>تاريخ الانضمام:</b> {user_info['join_date'][:10]}
💬 <b>عدد الرسائل:</b> {user_info['message_count']}
🚫 <b>محظور:</b> {'نعم' if user_info['is_blocked'] else 'لا'}
🕐 <b>آخر نشاط:</b> {user_info['last_active'][:16]}
"""
                    await bot.send_message(message.chat.id, info_text)
                else:
                    await bot.send_message(message.chat.id, "❌ لم يتم العثور على المستخدم")
            elif action == 'block':
                db.block_user(target_user_id)
                await bot.send_message(message.chat.id, f"✅ تم حظر المستخدم <code>{target_user_id}</code>")
            elif action == 'unblock':
                db.unblock_user(target_user_id)
                await bot.send_message(message.chat.id, f"✅ تم إلغاء حظر المستخدم <code>{target_user_id}</code>")
        except ValueError:
            await bot.send_message(message.chat.id, "❌ معرف المستخدم يجب أن يكون رقماً")
        return
    
    await bot.send_chat_action(message.chat.id, 'typing')
    db.update_user_activity(user_id)
    
    try:
        # الحصول على الرد من Multi-AI
        response, ai_name, ai_icon = await ai.get_response(text, user_name)
        
        formatted_response = format_code_response(response)
        
        # إضافة توقيع AI المستخدم
        formatted_response += f"\n\n━━━━━━━━━━━━━━\n<i>{ai_icon} Powered by {ai_name}</i>"
        
        db.save_conversation(user_id, text, response, ai_name)
        
        try:
            await bot.send_message(message.chat.id, formatted_response, parse_mode='HTML', disable_web_page_preview=True)
        except:
            await bot.send_message(message.chat.id, response + f"\n\n━━━━━━━━━━━━━━\n{ai_icon} Powered by {ai_name}", disable_web_page_preview=True)
    
    except Exception as e:
        logger.error(f"خطأ في معالجة الرسالة: {e}")
        await bot.send_message(message.chat.id, "❌ <b>عذراً!</b>\n\nحدث خطأ في معالجة طلبك. الرجاء المحاولة مرة أخرى.")

@bot.message_handler(content_types=['photo', 'video', 'document', 'audio', 'voice'])
async def handle_media_message(message):
    user_id = message.from_user.id
    if user_id in ADMIN_IDS and user_id in broadcast_mode:
        del broadcast_mode[user_id]
        await broadcast_message(message, user_id)
        return
    await bot.send_message(message.chat.id, "📎 <b>وسائط</b>\n\nحالياً، البوت يدعم الرسائل النصية فقط.\n\nالرجاء إرسال سؤالك كنص. 💬")

# تشغيل البوت
async def main():
    logger.info("🚀 بدء تشغيل Seno AI Bot...")
    logger.info(f"🤖 عدد الـ APIs المفعلة: {len(ai.apis)}")
    logger.info(f"📊 عدد المستخدمين المسجلين: {len(db.get_all_users())}")
    try:
        await bot.infinity_polling(timeout=60, long_polling_timeout=60, skip_pending=True)
    except Exception as e:
        logger.error(f"خطأ في تشغيل البوت: {e}")

if __name__ == '__main__':
    asyncio.run(main())
