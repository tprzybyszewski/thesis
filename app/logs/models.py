from django.db import models

class ConversationSession(models.Model):
    original_log = models.TextField() 
    summarized_log = models.TextField(blank=True) 
    conversation_history = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Conversation started on {self.created_at}"

