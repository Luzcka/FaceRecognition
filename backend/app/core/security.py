# core/security.py

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from core.config import settings

security = HTTPBearer()

class SecurityService:
    """
    Serviço de segurança da aplicação.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def validate_api_key(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
        """
        Valida a chave de API fornecida no header Authorization.
        
        Args:
            credentials: Credenciais de autenticação
            
        Returns:
            bool: True se a chave é válida
            
        Raises:
            HTTPException: Se a chave é inválida
        """
        if not credentials or credentials.credentials != self.api_key:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return True

# Instância global do serviço de segurança
security_service = SecurityService(settings.api_key)
