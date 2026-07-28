from typing import Union, List
from pydantic import BaseModel, Field

class EmailInput(BaseModel):
    body: str
    sender: str = ""
    recipient: str = ""
    subject: str = ""
    urls: List[str] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list)

class EmailPreprocessor:
    """Przygotowuje wejście do reprezentacji tekstowej dla embeddingu."""
    
    @staticmethod
    def process(data: Union[EmailInput, str]) -> str:
        if isinstance(data, str):
            return data.strip()
        
        # Jeśli podano samą treść bez nagłówków, zwracamy czysty tekst
        if not data.sender and not data.subject and not data.urls and not data.attachments:
            return data.body.strip()
            
        # Pełny format w przypadku przekazania metadanych
        urls_str = ", ".join(data.urls) if data.urls else "none"
        attachments_str = ", ".join(data.attachments) if data.attachments else "none"
        
        return (
            f"Sender: {data.sender.strip()} | "
            f"Subject: {data.subject.strip()} | "
            f"Body: {data.body.strip()} | "
            f"URLs: {urls_str} | "
            f"Attachments: {attachments_str}"
        )