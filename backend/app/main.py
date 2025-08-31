# main.py

# Para executar: uvicorn main:app --reload
# Para acessar Swagger: http://127.0.0.1:8000/docs
# ou 
# Executar definindo a porta: uvicorn main:app --reload --port 8010
# Para acessar Swagger: http://127.0.0.1:8010/docs


from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from api.endpoints import (register, 
                           search, 
                           tools)
from core.config import settings
from core.version import (__version__,
                          __description__,
                          show_version_info)


# Obter o diretório onde o script está localizado
SCRIPT_DIR =  Path(__file__).parent

# Configuração de logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Criar diretório de logs se não existir
log_dir = SCRIPT_DIR / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "app.log"

# Adicionar handler de arquivo
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

logger.info(f"Logs serão salvos em: {log_file.absolute()}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    logger.info("Iniciando aplicação de reconhecimento facial")
    logger.info(
        f"\nConfigurações: \n"
        f"  Modo={settings.milvus_mode},\n"
        f"  Modelo={settings.face_model},\n"
        f"  Threshold={settings.similarity_threshold}\n"
    )

    yield
    
    logger.info("Encerrando aplicação")

app = FastAPI(
    title="Facial Recognition PoC",
    version=__version__,
    description=__description__,
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção deve-se especificar os domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de tratamento de erros
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Tratamento global de exceções."""
    logger.error(f"Erro não tratado: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor"}
    )

# Rotas

ROOT = "/api/v1"
app.include_router(register.router, prefix=ROOT, tags=["Registration"])
app.include_router(search.router, prefix=ROOT, tags=["Search"])
app.include_router(tools.router, prefix=ROOT, tags=["Tools"])

@app.get("/")
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "message": __description__,
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Endpoint de health check da aplicação."""
    return {
        "status": "healthy",
        "model": settings.face_model,
        "dimension": settings.embedding_dimension,
        "threshold": settings.similarity_threshold
    }