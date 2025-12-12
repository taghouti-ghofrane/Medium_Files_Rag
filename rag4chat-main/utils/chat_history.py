"""
Chat history manager
"""
import json
import os
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
from config.settings import HISTORY_FILE, MAX_HISTORY_TURNS

class ChatHistoryManager:
    """
    Chat history manager class
    """
    def __init__(self):
        """Initialize chat history manager"""
        self.history: List[Dict] = self.load_history()
    
    # 1. Load history from file
    def load_history(self) -> List[Dict]:
        """
        Returns:
            List[Dict]: history list
        """
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading conversation history: {str(e)}")
        return []
    
    # 2. Save history to file
    def save_history(self) -> None:
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving conversation history: {str(e)}")
    
    # 3. Add message
    def add_message(self, role: str, content: str) -> None:
        """
        Args:
            role (str): 'user' or 'assistant'
            content (str): message content
        """
        self.history.append({"role": role, "content": content})
        self.save_history()
    
    # 4. Clear history
    def clear_history(self) -> None:
        self.history = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    
    # 5. Get formatted history
    def get_formatted_history(self, max_turns: int = MAX_HISTORY_TURNS) -> str:
        """
        Args:
            max_turns (int): max turns to keep
            
        Returns:
            str: formatted history
        """
        if not self.history:
            return ""
        
        recent_history = self.history[-max_turns*2:] if len(self.history) > max_turns*2 else self.history
        
        formatted_history = "Previous conversation:\n"
        for msg in recent_history:
            if msg["role"] == "user":
                role = "User"
            elif msg["role"] == "assistant":
                role = "Assistant"
            else:
                continue
            
            formatted_history += f"{role}: {msg['content']}\n"
        
        return formatted_history
    
    # 6. Export to CSV
    def export_to_csv(self) -> Optional[bytes]:
        """
        Export history to CSV
        
        Returns:
            Optional[bytes]: CSV content or None
        """
        try:
            df = pd.DataFrame(self.history)
            return df.to_csv(index=False).encode('utf-8')
        except Exception as e:
            print(f"Error exporting conversation history: {str(e)}")
            return None
    
    # 7. Get history stats
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics
        
        Returns:
            Dict[str, int]: counts
        """
        total_messages = len(self.history)
        user_messages = sum(1 for msg in self.history if msg["role"] == "user")
        return {
            "total_messages": total_messages,
            "user_messages": user_messages
        } 