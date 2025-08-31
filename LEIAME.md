# Sistema de Reconhecimento Facial - API

Uma API de reconhecimento facial de alta performance construída com DeepFace e Milvus para operações de busca por similaridade.

## Visão Geral

Este sistema fornece uma API REST para reconhecimento facial com capacidades de registro e busca de usuários. Utiliza modelos de deep learning de última geração para extração de embeddings faciais e busca de similaridade vetorial para recuperação rápida.

## Principais Funcionalidades

- Extração de embeddings faciais usando DeepFace com modelo Facenet512
- Busca de similaridade vetorial com banco de dados Milvus
- API REST com framework FastAPI
- Autenticação por Bearer token
- Sistema completo de logs e tratamento de erros
- Suporte à containerização Docker
- Thresholds de similaridade configuráveis

## Especificações Técnicas

- **Modelo Facial**: Facenet512 (embeddings de 512 dimensões)
- **Métrica de Distância**: Similaridade coseno
- **Banco Vetorial**: Milvus v2.5.x
- **Framework Web**: FastAPI
- **Autenticação**: Bearer token
- **Processamento de Imagem**: DeepFace e Detector backend OpenCV

## Estrutura do Projeto

```
facial-recognition-api/
├── api/
│   ├── dependencies.py      # Injeção de dependências
│   └── endpoints/
│       ├── register.py      # Endpoint de registro de usuário
│       └── search.py        # Endpoint de busca facial
├── core/
│   ├── config.py           # Configurações da aplicação
│   └── security.py         # Autenticação e segurança
├── models/
│   └── user.py             # Modelos de dados Pydantic
├── services/
│   ├── face_service.py     # Serviço de reconhecimento facial
│   └── milvus_service.py   # Serviço de banco vetorial
├── .env                    # Variáveis de ambiente
├── main.py                 # Aplicação principal
├── requirements.txt        # Dependências Python
├── Dockerfile             # Configuração Docker
└── docker-compose.yml     # Orquestração de containers
```

## Instalação

### Método 1: Ambiente Conda

```bash
# Crie ambiente conda
conda create -n facial-recognition python=3.9
conda activate facial-recognition

# Instale dependências
pip install -r requirements.txt

# Inicie o Milvus

# Inicie a aplicação
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Configuração

### Variáveis de Ambiente (.env)

```env
# Configuração da API
API_KEY=sua-chave-api-aqui
SECRET_KEY=sua-chave-secreta-aqui

# Configuração Milvus
MILVUS_MODE=remote
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_LOCAL_PATH=data/milvus_faces.db

# Configuração Reconhecimento Facial
FACE_MODEL=Facenet512
FACE_DETECTOR=opencv
SIMILARITY_THRESHOLD=0.3
TOP_K_RESULTS=5

# Logging
LOG_LEVEL=INFO
```

### Configuração do Milvus

Para desenvolvimento local, você pode executar Milvus com Docker:


#### Método 1
```bash
# Utilize o docker compose localizado em: milvus_docker_compose 

# Crie a pasta para o docker compose
mkdir -p ~/milvus-standalone
cd ~/milvus-standalone

# copie o arquivo docker-compose.yml para o diretório criado

sudo docker compose up -d

# Alterar o proprietário dos volumes
sudo chown -R $USER:$USER volumes
sudo chown -R 1000:1000 volumes/milvus volumes/minio volumes/etcd
```

#### Método 2
```bash
# Baixe e execute Milvus standalone
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

## Uso da API

### Autenticação

Todos os endpoints requerem autenticação Bearer token:

```
Authorization: Bearer sua-chave-api-aqui
```

### Registrar Usuário

```bash
curl -X POST "http://localhost:8000/api/v1/register" \
  -H "Authorization: Bearer sua-chave-api-aqui" \
  -F "name=João Silva" \
  -F "registration_number=EMP001" \
  -F "image=@pessoa.jpg"
```

### Buscar Usuário

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Authorization: Bearer sua-chave-api-aqui" \
  -F "image=@busca.jpg"
```

### Verificação de Saúde

```bash
curl http://localhost:8000/health
```

## Exemplos de Resposta da API

### Busca Bem-sucedida

```json
[
  {
    "name": "João Silva",
    "registration_number": "EMP001",
    "similarity_score": 0.987,
    "distance": 0.013
  },
  {
    "name": "Maria Santos",
    "registration_number": "EMP002",
    "similarity_score": 0.956,
    "distance": 0.044
  }
]
```

### Registro Bem-sucedido

```json
{
  "message": "Usuário registrado com sucesso",
  "name": "João Silva",
  "registration_number": "EMP001"
}
```

## Desenvolvimento

### Sistema de Logs

Os logs são escritos no diretório `backend/app/logs/`. Configure o nível de log no `.env`:

- DEBUG: Informações detalhadas para depuração
- INFO: Informações gerais sobre operação do sistema
- WARN: Mensagens de aviso
- ERROR: Condições de erro
- CRITICAL: Condições de erro crítico

## Solução de Problemas

### Problemas Comuns

1. **Erro de Conexão Milvus**: Certifique-se de que o servidor Milvus está rodando e acessível
2. **Face Não Detectada**: Verifique a qualidade da imagem e condições de iluminação
3. **Scores de Similaridade Baixos**: Ajuste SIMILARITY_THRESHOLD na configuração
4. **Problemas de Memória**: Monitore recursos dos containers Docker

### Informações de Debug

Acesse o endpoint de debug para informações do sistema:

```bash
curl -H "Authorization: Bearer sua-chave-api-aqui" \
  http://localhost:8000/debug/info
```

## Requisitos

- Python 3.11+
- Docker (opcional)
- Milvus v2.5.16
- Mínimo 4GB RAM
- Suporte GPU (opcional, para processamento mais rápido)

## Licença

Uso apenas baixo autorização