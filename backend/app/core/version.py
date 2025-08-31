# Controle de versão principal
# Ao migrar para Git é interessante utilizar uma das seguintes opçÕes:
#   - setuptools_scm (pip install setuptools_scm)
#   - python-semantic-release (pip install python-semantic-release)

import subprocess
import os


__version_info__ = (0, 5, 0)  # major, minor, patch
__version__ = ".".join(str(v) for v in __version_info__)

# Informações adicionais
__build_date__ = "2025-07-17"
__description__ = "Sistema de reconhecimento facial com busca por similaridade"

# Tenta obter informações do Git se disponível
try:
    if os.path.exists(".git"):
        # Obtém o hash do commit atual
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        # Verifica se há alterações não commitadas
        git_dirty = len(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).strip()) > 0
        
        __version__ = f"{__version__}+{git_commit}{'dirty' if git_dirty else ''}"
except (subprocess.SubprocessError, OSError) as e:
    # Falha silenciosamente se Git não estiver disponível
    pass

# Função para exibir informações da versão
def show_version_info():
    """Retorna uma string formatada com informações completas da versão"""
    return f"Facial Recognition PoC v{__version__} ({__build_date__})"