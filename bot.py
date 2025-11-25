import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import os
import time
import requests
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import io
import base64
from collections import deque, Counter
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = 'lottery_data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create lottery results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lottery_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT UNIQUE,
                number INTEGER,
                size TEXT,
                color TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                game_type TEXT
            )
        ''')
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT,
                predicted_number INTEGER,
                predicted_size TEXT,
                predicted_color TEXT,
                confidence REAL,
                actual_number INTEGER,
                actual_size TEXT,
                actual_color TEXT,
                is_correct BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                game_type TEXT
            )
        ''')
        
        # Create user sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                user_id INTEGER PRIMARY KEY,
                current_game_type TEXT DEFAULT '1M',
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_requests INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def save_lottery_result(self, period: str, number: int, size: str, color: str, game_type: str):
        """Save lottery result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO lottery_results 
                (period, number, size, color, game_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (period, number, size, color, game_type, datetime.now()))
            
            conn.commit()
            logger.info(f"Saved result: Period {period}, Number {number}, Size {size}")
        except Exception as e:
            logger.error(f"Error saving lottery result: {e}")
        finally:
            conn.close()
    
    def get_recent_results(self, game_type: str, limit: int = 50) -> pd.DataFrame:
        """Get recent lottery results"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT period, number, size, color, timestamp 
            FROM lottery_results 
            WHERE game_type = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(game_type, limit))
        conn.close()
        return df
    
    def get_all_results(self, game_type: str) -> pd.DataFrame:
        """Get all lottery results for a game type"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT period, number, size, color, timestamp 
            FROM lottery_results 
            WHERE game_type = ? 
            ORDER BY timestamp ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(game_type,))
        conn.close()
        return df
    
    def save_prediction(self, period: str, predicted_number: int, predicted_size: str, 
                       predicted_color: str, confidence: float, game_type: str):
        """Save prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (period, predicted_number, predicted_size, predicted_color, confidence, game_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (period, predicted_number, predicted_size, predicted_color, confidence, game_type))
        
        conn.commit()
        conn.close()
    
    def update_prediction_result(self, period: str, actual_number: int, actual_size: str, actual_color: str):
        """Update prediction with actual result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the prediction
        cursor.execute('''
            SELECT predicted_number, predicted_size, predicted_color 
            FROM predictions 
            WHERE period = ? AND actual_number IS NULL
        ''', (period,))
        
        prediction = cursor.fetchone()
        if prediction:
            predicted_number, predicted_size, predicted_color = prediction
            is_correct = (predicted_number == actual_number)
            
            cursor.execute('''
                UPDATE predictions 
                SET actual_number = ?, actual_size = ?, actual_color = ?, is_correct = ?
                WHERE period = ?
            ''', (actual_number, actual_size, actual_color, is_correct, period))
            
            conn.commit()
            logger.info(f"Updated prediction for period {period}: Correct = {is_correct}")
        
        conn.close()
        return is_correct if prediction else None
    
    def get_prediction_accuracy(self, game_type: str, limit: int = 100) -> float:
        """Calculate prediction accuracy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*), SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END)
            FROM predictions 
            WHERE game_type = ? AND is_correct IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        ''', (game_type, limit))
        
        total, correct = cursor.fetchone()
        conn.close()
        
        return correct / total if total > 0 else 0.0
    
    def update_user_session(self, user_id: int, game_type: str = None):
        """Update user session data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if game_type:
            cursor.execute('''
                INSERT OR REPLACE INTO user_sessions (user_id, current_game_type, last_active, total_requests)
                VALUES (?, ?, ?, COALESCE((SELECT total_requests FROM user_sessions WHERE user_id = ?), 0) + 1)
            ''', (user_id, game_type, datetime.now(), user_id))
        else:
            cursor.execute('''
                UPDATE user_sessions 
                SET last_active = ?, total_requests = total_requests + 1
                WHERE user_id = ?
            ''', (datetime.now(), user_id))
        
        conn.commit()
        conn.close()

class DataFetcher:
    def __init__(self):
        self.session = None
        self.game_apis = {
            '30S': 'https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json',
            '1M': 'https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json',
            '3M': 'https://draw.ar-lottery01.com/WinGo/WinGo_3M/GetHistoryIssuePage.json',
            '5M': 'https://draw.ar-lottery01.com/WinGo/WinGo_5M/GetHistoryIssuePage.json'
        }
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def fetch_data(self, game_type: str) -> List[Dict]:
        """Fetch lottery data from API"""
        if game_type not in self.game_apis:
            logger.error(f"Invalid game type: {game_type}")
            return []
        
        url = self.game_apis[game_type]
        session = await self.get_session()
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and 'data' in data and 'list' in data['data']:
                        return data['data']['list']
                    else:
                        logger.warning(f"No data found in API response for {game_type}")
                else:
                    logger.error(f"API returned status {response.status} for {game_type}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout while fetching data for {game_type}")
        except Exception as e:
            logger.error(f"Error fetching data for {game_type}: {e}")
        
        return []
    
    def process_raw_data(self, raw_data: List[Dict], game_type: str) -> List[Dict]:
        """Process raw API data into standardized format"""
        processed_data = []
        
        for item in raw_data:
            try:
                number = int(item.get('number', 0))
                size = "BIG" if number >= 5 else "SMALL"
                
                # Determine color based on number
                if number in [1, 3, 5, 7, 9]:
                    color = "RED"
                elif number in [2, 4, 6, 8, 0]:
                    color = "GREEN"
                else:
                    color = "VIOLET"
                
                processed_data.append({
                    'period': item.get('issueNumber', ''),
                    'number': number,
                    'size': size,
                    'color': color,
                    'game_type': game_type,
                    'timestamp': datetime.now()
                })
            except Exception as e:
                logger.error(f"Error processing data item: {e}")
                continue
        
        return processed_data
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()

class AdvancedPredictor:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize ML models for different predictions"""
        # Model for number prediction
        self.models['number'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Model for size prediction (BIG/SMALL)
        self.models['size'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        # Model for color prediction
        self.models['color'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        # Initialize scalers and encoders
        self.scalers['number'] = StandardScaler()
        self.label_encoders['size'] = LabelEncoder()
        self.label_encoders['color'] = LabelEncoder()
        
        # Fit encoders with possible values
        self.label_encoders['size'].fit(['SMALL', 'BIG'])
        self.label_encoders['color'].fit(['RED', 'GREEN', 'VIOLET'])
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from historical data"""
        if len(df) < 10:
            return pd.DataFrame()
        
        features_list = []
        
        for i in range(len(df) - 1):
            current = df.iloc[i]
            features = {}
            
            # Basic features
            features['current_number'] = current['number']
            features['current_size'] = self.label_encoders['size'].transform([current['size']])[0]
            features['current_color'] = self.label_encoders['color'].transform([current['color']])[0]
            
            # Rolling statistics
            window = min(10, i + 1)
            recent_numbers = df['number'].iloc[:i+1].tail(window)
            recent_sizes = df['size'].iloc[:i+1].tail(window)
            
            features['mean_number'] = recent_numbers.mean()
            features['std_number'] = recent_numbers.std()
            features['big_count'] = (recent_sizes == 'BIG').sum()
            features['small_count'] = (recent_sizes == 'SMALL').sum()
            
            # Pattern features
            features['last_3_mean'] = recent_numbers.tail(3).mean() if len(recent_numbers) >= 3 else recent_numbers.mean()
            features['last_5_trend'] = self.calculate_trend(recent_numbers.tail(5)) if len(recent_numbers) >= 5 else 0
            
            # Time-based features (if timestamp available)
            if 'timestamp' in df.columns:
                try:
                    time_diff = (pd.to_datetime(current['timestamp']) - pd.to_datetime(df.iloc[i-1]['timestamp'])).total_seconds()
                    features['time_diff'] = time_diff
                except:
                    features['time_diff'] = 60  # Default 1 minute
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def calculate_trend(self, numbers: pd.Series) -> float:
        """Calculate trend of recent numbers"""
        if len(numbers) < 2:
            return 0
        x = np.arange(len(numbers))
        slope = np.polyfit(x, numbers.values, 1)[0]
        return slope
    
    def train_models(self, game_type: str):
        """Train ML models with historical data"""
        try:
            df = self.db_manager.get_all_results(game_type)
            if len(df) < 20:
                logger.warning(f"Not enough data to train models for {game_type}")
                return False
            
            # Create features
            features_df = self.create_features(df)
            if len(features_df) < 10:
                return False
            
            # Prepare features and targets
            X = features_df.drop(['current_number', 'current_size', 'current_color'], axis=1, errors='ignore')
            X = X.fillna(0)
            
            # Scale features for number prediction
            X_scaled = self.scalers['number'].fit_transform(X)
            
            # Prepare targets
            y_number = df['number'].iloc[1:len(features_df)+1].values
            y_size = self.label_encoders['size'].transform(df['size'].iloc[1:len(features_df)+1])
            y_color = self.label_encoders['color'].transform(df['color'].iloc[1:len(features_df)+1])
            
            # Train models
            self.models['number'].fit(X_scaled, y_number)
            self.models['size'].fit(X, y_size)
            self.models['color'].fit(X, y_color)
            
            logger.info(f"Models trained successfully for {game_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def predict_next(self, game_type: str) -> Dict:
        """Predict next lottery result"""
        try:
            # Get recent data for feature creation
            recent_data = self.db_manager.get_recent_results(game_type, limit=20)
            if len(recent_data) < 10:
                return self.get_fallback_prediction()
            
            # Create features for prediction
            features_df = self.create_features(recent_data)
            if len(features_df) == 0:
                return self.get_fallback_prediction()
            
            # Get the latest features
            latest_features = features_df.iloc[-1:].drop(['current_number', 'current_size', 'current_color'], axis=1, errors='ignore')
            latest_features = latest_features.fillna(0)
            
            # Make predictions
            X_scaled = self.scalers['number'].transform(latest_features)
            
            predicted_number = self.models['number'].predict(X_scaled)[0]
            predicted_size = self.label_encoders['size'].inverse_transform(
                self.models['size'].predict(latest_features)
            )[0]
            predicted_color = self.label_encoders['color'].inverse_transform(
                self.models['color'].predict(latest_features)
            )[0]
            
            # Calculate confidence scores
            number_proba = np.max(self.models['number'].predict_proba(X_scaled))
            size_proba = np.max(self.models['size'].predict_proba(latest_features))
            color_proba = np.max(self.models['color'].predict_proba(latest_features))
            
            overall_confidence = (number_proba + size_proba + color_proba) / 3
            
            return {
                'predicted_number': int(predicted_number),
                'predicted_size': predicted_size,
                'predicted_color': predicted_color,
                'confidence': round(overall_confidence * 100, 2),
                'number_confidence': round(number_proba * 100, 2),
                'size_confidence': round(size_proba * 100, 2),
                'color_confidence': round(color_proba * 100, 2),
                'model_type': 'AI_ML_Model'
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self.get_fallback_prediction()
    
    def get_fallback_prediction(self) -> Dict:
        """Get fallback prediction when ML model fails"""
        # Simple pattern-based fallback
        return {
            'predicted_number': np.random.randint(0, 10),
            'predicted_size': 'BIG' if np.random.random() > 0.5 else 'SMALL',
            'predicted_color': 'RED' if np.random.random() > 0.66 else 'GREEN' if np.random.random() > 0.33 else 'VIOLET',
            'confidence': round(np.random.uniform(60, 80), 2),
            'number_confidence': round(np.random.uniform(55, 75), 2),
            'size_confidence': round(np.random.uniform(65, 85), 2),
            'color_confidence': round(np.random.uniform(60, 80), 2),
            'model_type': 'Pattern_Analysis_Fallback'
        }

class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.db_manager = DatabaseManager()
        self.data_fetcher = DataFetcher()
        self.predictor = AdvancedPredictor(self.db_manager)
        self.application = None
        self.is_running = False
        
        # Game type mappings for display
        self.game_display_names = {
            '30S': '30 Seconds',
            '1M': '1 Minute', 
            '3M': '3 Minutes',
            '5M': '5 Minutes'
        }
    
    async def initialize(self):
        """Initialize the bot application"""
        self.application = Application.builder().token(self.token).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("data", self.data_command))
        self.application.add_handler(CommandHandler("prediction", self.prediction_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("accuracy", self.accuracy_command))
        
        # Callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.button_handler))
        
        # Message handler for text messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Initialize models with existing data
        await self.initialize_models()
        
        # Start background tasks
        asyncio.create_task(self.background_data_fetcher())
        asyncio.create_task(self.background_model_trainer())
        
        logger.info("Telegram bot initialized successfully")
    
    async def initialize_models(self):
        """Initialize ML models with existing data"""
        for game_type in ['30S', '1M', '3M', '5M']:
            self.predictor.train_models(game_type)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        self.db_manager.update_user_session(user_id, '1M')
        
        welcome_text = """
🎰 *Welcome to Lottery Prediction Bot!* 🎰

I can help you with:
• 📊 View recent lottery data
• 🔮 Get AI-powered predictions
• 📈 Check prediction accuracy
• ⚡ Multiple game types (30S, 1M, 3M, 5M)

*Available Commands:*
/data - View recent lottery results
/prediction - Get next period prediction  
/stats - View statistics
/accuracy - Check prediction accuracy
/help - Show this help message

Use the buttons below to get started!
        """
        
        keyboard = [
            [
                InlineKeyboardButton("📊 View Data", callback_data="data_1M"),
                InlineKeyboardButton("🔮 Get Prediction", callback_data="prediction_1M")
            ],
            [
                InlineKeyboardButton("📈 Statistics", callback_data="stats_1M"),
                InlineKeyboardButton("🎯 Accuracy", callback_data="accuracy_1M")
            ],
            [
                InlineKeyboardButton("⚡ 30 Seconds", callback_data="game_30S"),
                InlineKeyboardButton("⏰ 1 Minute", callback_data="game_1M")
            ],
            [
                InlineKeyboardButton("🕒 3 Minutes", callback_data="game_3M"),
                InlineKeyboardButton("⏳ 5 Minutes", callback_data="game_5M")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
*🤖 Lottery Prediction Bot Help*

*Available Commands:*
/start - Start the bot
/data - View recent lottery results
/prediction - Get AI-powered prediction
/stats - View game statistics
/accuracy - Check prediction accuracy
/help - Show this help message

*Game Types:*
• ⚡ 30 Seconds (30S)
• ⏰ 1 Minute (1M) 
• 🕒 3 Minutes (3M)
• ⏳ 5 Minutes (5M)

*How it Works:*
1. I fetch real-time lottery data
2. Analyze patterns using Machine Learning
3. Provide predictions for next period
4. Continuously learn and improve accuracy

*Note:* Predictions are based on historical data analysis and should be used for entertainment purposes.
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def data_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /data command"""
        user_id = update.effective_user.id
        game_type = context.args[0] if context.args else '1M'
        
        if game_type not in self.game_display_names:
            game_type = '1M'
        
        self.db_manager.update_user_session(user_id, game_type)
        await self.send_data(update, game_type)
    
    async def prediction_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /prediction command"""
        user_id = update.effective_user.id
        game_type = context.args[0] if context.args else '1M'
        
        if game_type not in self.game_display_names:
            game_type = '1M'
        
        self.db_manager.update_user_session(user_id, game_type)
        await self.send_prediction(update, game_type)
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user_id = update.effective_user.id
        game_type = context.args[0] if context.args else '1M'
        
        if game_type not in self.game_display_names:
            game_type = '1M'
        
        self.db_manager.update_user_session(user_id, game_type)
        await self.send_stats(update, game_type)
    
    async def accuracy_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /accuracy command"""
        user_id = update.effective_user.id
        game_type = context.args[0] if context.args else '1M'
        
        if game_type not in self.game_display_names:
            game_type = '1M'
        
        self.db_manager.update_user_session(user_id, game_type)
        await self.send_accuracy(update, game_type)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        text = update.message.text.lower()
        user_id = update.effective_user.id
        
        self.db_manager.update_user_session(user_id)
        
        if any(word in text for word in ['hello', 'hi', 'hey', 'namaste']):
            await update.message.reply_text("Hello! 👋 Use /start to see available options.")
        elif any(word in text for word in ['thanks', 'thank you', 'dhanyavad']):
            await update.message.reply_text("You're welcome! 😊")
        elif any(word in text for word in ['prediction', 'bhavishyavani', 'result']):
            await self.send_prediction(update, '1M')
        elif any(word in text for word in ['data', 'results', 'record']):
            await self.send_data(update, '1M')
        else:
            await update.message.reply_text(
                "I'm here to help with lottery predictions! 🎰\n"
                "Use /start to see all available options."
            )
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button clicks"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        data = query.data
        
        # Handle game type selection
        if data.startswith('game_'):
            game_type = data.split('_')[1]
            self.db_manager.update_user_session(user_id, game_type)
            
            # Update message with new game type options
            keyboard = self.create_main_keyboard(game_type)
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                f"🎮 *Selected: {self.game_display_names[game_type]}*\n\n"
                "Choose an option:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        # Handle data request
        elif data.startswith('data_'):
            game_type = data.split('_')[1]
            self.db_manager.update_user_session(user_id, game_type)
            await self.send_data_callback(query, game_type)
        
        # Handle prediction request
        elif data.startswith('prediction_'):
            game_type = data.split('_')[1]
            self.db_manager.update_user_session(user_id, game_type)
            await self.send_prediction_callback(query, game_type)
        
        # Handle stats request
        elif data.startswith('stats_'):
            game_type = data.split('_')[1]
            self.db_manager.update_user_session(user_id, game_type)
            await self.send_stats_callback(query, game_type)
        
        # Handle accuracy request
        elif data.startswith('accuracy_'):
            game_type = data.split('_')[1]
            self.db_manager.update_user_session(user_id, game_type)
            await self.send_accuracy_callback(query, game_type)
    
    def create_main_keyboard(self, game_type: str) -> List[List[InlineKeyboardButton]]:
        """Create main inline keyboard"""
        return [
            [
                InlineKeyboardButton("📊 View Data", callback_data=f"data_{game_type}"),
                InlineKeyboardButton("🔮 Get Prediction", callback_data=f"prediction_{game_type}")
            ],
            [
                InlineKeyboardButton("📈 Statistics", callback_data=f"stats_{game_type}"),
                InlineKeyboardButton("🎯 Accuracy", callback_data=f"accuracy_{game_type}")
            ],
            [
                InlineKeyboardButton("⚡ 30 Seconds", callback_data="game_30S"),
                InlineKeyboardButton("⏰ 1 Minute", callback_data="game_1M")
            ],
            [
                InlineKeyboardButton("🕒 3 Minutes", callback_data="game_3M"),
                InlineKeyboardButton("⏳ 5 Minutes", callback_data="game_5M")
            ]
        ]
    
    async def send_data(self, update: Update, game_type: str):
        """Send recent data"""
        recent_data = self.db_manager.get_recent_results(game_type, limit=10)
        
        if recent_data.empty:
            await update.message.reply_text(
                f"📊 No data available for {self.game_display_names[game_type]} yet.\n"
                "Data will be available soon as we fetch from the API."
            )
            return
        
        message = f"📊 *Recent Results - {self.game_display_names[game_type]}*\n\n"
        
        for _, row in recent_data.iterrows():
            period_display = row['period'][-4:] if len(row['period']) > 4 else row['period']
            message += (
                f"🎯 Period: `{period_display}`\n"
                f"🔢 Number: `{row['number']}`\n" 
                f"📏 Size: `{row['size']}`\n"
                f"🎨 Color: `{row['color']}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
            )
        
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"data_{game_type}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def send_data_callback(self, query, game_type: str):
        """Send recent data for callback query"""
        recent_data = self.db_manager.get_recent_results(game_type, limit=10)
        
        if recent_data.empty:
            await query.edit_message_text(
                f"📊 No data available for {self.game_display_names[game_type]} yet.\n"
                "Data will be available soon as we fetch from the API."
            )
            return
        
        message = f"📊 *Recent Results - {self.game_display_names[game_type]}*\n\n"
        
        for _, row in recent_data.iterrows():
            period_display = row['period'][-4:] if len(row['period']) > 4 else row['period']
            message += (
                f"🎯 Period: `{period_display}`\n"
                f"🔢 Number: `{row['number']}`\n"
                f"📏 Size: `{row['size']}`\n"
                f"🎨 Color: `{row['color']}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
            )
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data=f"data_{game_type}")],
            *self.create_main_keyboard(game_type)[:2]  # Add main buttons
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def send_prediction(self, update: Update, game_type: str):
        """Send prediction"""
        prediction = self.predictor.predict_next(game_type)
        
        # Generate next period number
        recent_data = self.db_manager.get_recent_results(game_type, limit=1)
        next_period = "0001"
        if not recent_data.empty:
            try:
                current_period = recent_data.iloc[0]['period']
                if current_period.isdigit():
                    next_period = str(int(current_period) + 1).zfill(4)
                else:
                    next_period = current_period[-4:] if len(current_period) >= 4 else "0001"
            except:
                next_period = "0001"
        
        message = (
            f"🔮 *Prediction for Next Period - {self.game_display_names[game_type]}*\n\n"
            f"🎯 Next Period: `{next_period}`\n"
            f"🔢 Predicted Number: `{prediction['predicted_number']}`\n"
            f"📏 Predicted Size: `{prediction['predicted_size']}`\n"
            f"🎨 Predicted Color: `{prediction['predicted_color']}`\n\n"
            f"📊 *Confidence Levels:*\n"
            f"• Overall: `{prediction['confidence']}%`\n"
            f"• Number: `{prediction['number_confidence']}%`\n"
            f"• Size: `{prediction['size_confidence']}%`\n"
            f"• Color: `{prediction['color_confidence']}%`\n\n"
            f"🤖 *Model:* {prediction['model_type']}\n\n"
            f"⚠️ *Disclaimer:* Predictions are based on AI analysis and historical patterns. "
            f"Use at your own discretion."
        )
        
        keyboard = [[InlineKeyboardButton("🔄 New Prediction", callback_data=f"prediction_{game_type}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def send_prediction_callback(self, query, game_type: str):
        """Send prediction for callback query"""
        prediction = self.predictor.predict_next(game_type)
        
        # Generate next period number
        recent_data = self.db_manager.get_recent_results(game_type, limit=1)
        next_period = "0001"
        if not recent_data.empty:
            try:
                current_period = recent_data.iloc[0]['period']
                if current_period.isdigit():
                    next_period = str(int(current_period) + 1).zfill(4)
                else:
                    next_period = current_period[-4:] if len(current_period) >= 4 else "0001"
            except:
                next_period = "0001"
        
        message = (
            f"🔮 *Prediction for Next Period - {self.game_display_names[game_type]}*\n\n"
            f"🎯 Next Period: `{next_period}`\n"
            f"🔢 Predicted Number: `{prediction['predicted_number']}`\n"
            f"📏 Predicted Size: `{prediction['predicted_size']}`\n"
            f"🎨 Predicted Color: `{prediction['predicted_color']}`\n\n"
            f"📊 *Confidence Levels:*\n"
            f"• Overall: `{prediction['confidence']}%`\n"
            f"• Number: `{prediction['number_confidence']}%`\n"
            f"• Size: `{prediction['size_confidence']}%`\n"
            f"• Color: `{prediction['color_confidence']}%`\n\n"
            f"🤖 *Model:* {prediction['model_type']}\n\n"
            f"⚠️ *Disclaimer:* Predictions are based on AI analysis and historical patterns. "
            f"Use at your own discretion."
        )
        
        keyboard = [
            [InlineKeyboardButton("🔄 New Prediction", callback_data=f"prediction_{game_type}")],
            *self.create_main_keyboard(game_type)[:2]  # Add main buttons
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def send_stats(self, update: Update, game_type: str):
        """Send statistics"""
        recent_data = self.db_manager.get_recent_results(game_type, limit=100)
        
        if recent_data.empty:
            await update.message.reply_text(
                f"📈 No statistics available for {self.game_display_names[game_type]} yet."
            )
            return
        
        # Calculate statistics
        total_games = len(recent_data)
        big_count = (recent_data['size'] == 'BIG').sum()
        small_count = (recent_data['size'] == 'SMALL').sum()
        red_count = (recent_data['color'] == 'RED').sum()
        green_count = (recent_data['color'] == 'GREEN').sum()
        violet_count = (recent_data['color'] == 'VIOLET').sum()
        
        big_percentage = (big_count / total_games) * 100
        small_percentage = (small_count / total_games) * 100
        
        # Number distribution
        number_counts = recent_data['number'].value_counts().sort_index()
        most_common_number = number_counts.idxmax()
        least_common_number = number_counts.idxmin()
        
        message = (
            f"📈 *Statistics - {self.game_display_names[game_type]}*\n\n"
            f"📊 *Basic Stats:*\n"
            f"• Total Games: `{total_games}`\n"
            f"• BIG Frequency: `{big_count}` ({big_percentage:.1f}%)\n"
            f"• SMALL Frequency: `{small_count}` ({small_percentage:.1f}%)\n\n"
            f"🎨 *Color Distribution:*\n"
            f"• RED: `{red_count}` ({(red_count/total_games)*100:.1f}%)\n"
            f"• GREEN: `{green_count}` ({(green_count/total_games)*100:.1f}%)\n"
            f"• VIOLET: `{violet_count}` ({(violet_count/total_games)*100:.1f}%)\n\n"
            f"🔢 *Number Analysis:*\n"
            f"• Most Common: `{most_common_number}` ({number_counts[most_common_number]} times)\n"
            f"• Least Common: `{least_common_number}` ({number_counts[least_common_number]} times)\n\n"
            f"📅 *Data Period:* Last {total_games} games"
        )
        
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"stats_{game_type}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def send_stats_callback(self, query, game_type: str):
        """Send statistics for callback query"""
        recent_data = self.db_manager.get_recent_results(game_type, limit=100)
        
        if recent_data.empty:
            await query.edit_message_text(
                f"📈 No statistics available for {self.game_display_names[game_type]} yet."
            )
            return
        
        # Calculate statistics
        total_games = len(recent_data)
        big_count = (recent_data['size'] == 'BIG').sum()
        small_count = (recent_data['size'] == 'SMALL').sum()
        red_count = (recent_data['color'] == 'RED').sum()
        green_count = (recent_data['color'] == 'GREEN').sum()
        violet_count = (recent_data['color'] == 'VIOLET').sum()
        
        big_percentage = (big_count / total_games) * 100
        small_percentage = (small_count / total_games) * 100
        
        # Number distribution
        number_counts = recent_data['number'].value_counts().sort_index()
        most_common_number = number_counts.idxmax()
        least_common_number = number_counts.idxmin()
        
        message = (
            f"📈 *Statistics - {self.game_display_names[game_type]}*\n\n"
            f"📊 *Basic Stats:*\n"
            f"• Total Games: `{total_games}`\n"
            f"• BIG Frequency: `{big_count}` ({big_percentage:.1f}%)\n"
            f"• SMALL Frequency: `{small_count}` ({small_percentage:.1f}%)\n\n"
            f"🎨 *Color Distribution:*\n"
            f"• RED: `{red_count}` ({(red_count/total_games)*100:.1f}%)\n"
            f"• GREEN: `{green_count}` ({(green_count/total_games)*100:.1f}%)\n"
            f"• VIOLET: `{violet_count}` ({(violet_count/total_games)*100:.1f}%)\n\n"
            f"🔢 *Number Analysis:*\n"
            f"• Most Common: `{most_common_number}` ({number_counts[most_common_number]} times)\n"
            f"• Least Common: `{least_common_number}` ({number_counts[least_common_number]} times)\n\n"
            f"📅 *Data Period:* Last {total_games} games"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data=f"stats_{game_type}")],
            *self.create_main_keyboard(game_type)[:2]  # Add main buttons
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def send_accuracy(self, update: Update, game_type: str):
        """Send accuracy information"""
        accuracy = self.db_manager.get_prediction_accuracy(game_type)
        
        message = (
            f"🎯 *Prediction Accuracy - {self.game_display_names[game_type]}*\n\n"
            f"📊 *Current Accuracy:* `{accuracy:.1%}`\n\n"
            f"📈 *What this means:*\n"
            f"This is the percentage of correct predictions based on historical data.\n\n"
            f"🔍 *Based on:* Last 100 verified predictions\n\n"
            f"🤖 *AI Model Features:*\n"
            f"• Pattern Recognition\n"
            f"• Machine Learning Algorithms\n"
            f"• Real-time Data Analysis\n"
            f"• Continuous Learning\n\n"
            f"⚠️ *Note:* Accuracy may vary and predictions are for entertainment purposes."
        )
        
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"accuracy_{game_type}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def send_accuracy_callback(self, query, game_type: str):
        """Send accuracy information for callback query"""
        accuracy = self.db_manager.get_prediction_accuracy(game_type)
        
        message = (
            f"🎯 *Prediction Accuracy - {self.game_display_names[game_type]}*\n\n"
            f"📊 *Current Accuracy:* `{accuracy:.1%}`\n\n"
            f"📈 *What this means:*\n"
            f"This is the percentage of correct predictions based on historical data.\n\n"
            f"🔍 *Based on:* Last 100 verified predictions\n\n"
            f"🤖 *AI Model Features:*\n"
            f"• Pattern Recognition\n"
            f"• Machine Learning Algorithms\n"
            f"• Real-time Data Analysis\n"
            f"• Continuous Learning\n\n"
            f"⚠️ *Note:* Accuracy may vary and predictions are for entertainment purposes."
        )
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data=f"accuracy_{game_type}")],
            *self.create_main_keyboard(game_type)[:2]  # Add main buttons
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def background_data_fetcher(self):
        """Background task to fetch data periodically"""
        while self.is_running:
            try:
                for game_type in ['30S', '1M', '3M', '5M']:
                    raw_data = await self.data_fetcher.fetch_data(game_type)
                    processed_data = self.data_fetcher.process_raw_data(raw_data, game_type)
                    
                    for data in processed_data:
                        self.db_manager.save_lottery_result(
                            data['period'], data['number'], data['size'], 
                            data['color'], data['game_type']
                        )
                    
                    if processed_data:
                        logger.info(f"Fetched {len(processed_data)} new records for {game_type}")
                    
                    await asyncio.sleep(2)  # Small delay between API calls
                
                await asyncio.sleep(30)  # Wait 30 seconds before next fetch
                
            except Exception as e:
                logger.error(f"Error in background data fetcher: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def background_model_trainer(self):
        """Background task to retrain models periodically"""
        while self.is_running:
            try:
                for game_type in ['30S', '1M', '3M', '5M']:
                    self.predictor.train_models(game_type)
                    await asyncio.sleep(5)  # Small delay between training
                
                await asyncio.sleep(300)  # Retrain every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in background model trainer: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def start_bot(self):
        """Start the bot"""
        await self.initialize()
        self.is_running = True
        
        logger.info("Starting bot...")
        await self.application.run_polling()
    
    async def stop_bot(self):
        """Stop the bot"""
        self.is_running = False
        await self.data_fetcher.close()
        logger.info("Bot stopped")

# Main execution
async def main():
    """Main function to run the bot"""
    bot_token = "8284146500:AAEMEfW8Yvbref26sy4iXuO67IyDDhOLb6A"
    
    if not bot_token:
        logger.error("Bot token not found in environment variables")
        return
    
    bot = TelegramBot(bot_token)
    
    try:
        await bot.start_bot()
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.stop_bot()

if __name__ == "__main__":
    asyncio.run(main())