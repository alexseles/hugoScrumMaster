# Arquitetura do Agente Copilot para Scrum Masters

## Visão Geral

Este projeto implementa um agente de IA especializado para auxiliar Scrum Masters, combinando o poder de Grandes Modelos de Linguagem (LLM) com a técnica RAG (Geração Aumentada por Recuperação). O agente fornece suporte contextualizado com base em uma base de conhecimento que inclui o Guia do Scrum, o Manifesto Ágil e outros documentos relevantes sobre metodologias ágeis.

## Arquitetura do Sistema

```text
┌─────────────────┐      HTTP/JSON API      ┌─────────────────────────────────┐
│    Frontend     │ <---------------------->│         Backend (Python)        │
│ (HTML/CSS/JS)   │                         │             (Flask)             │
└─────────────────┘                         └─────────────────────────────────┘
        │                                                   │
        ▼                                                   ▼
┌─────────────────┐                         ┌─────────────────────────────────┐
│ Interface do    │                         │  Sistema RAG + LLM Integration  │
│    Usuário      │                         │  (LangChain, FAISS, Groq/LLM)   │
└─────────────────┘                         └─────────────────────────────────┘

scrum-copilot/
│
├── app.py                     # Lógica do backend (Flask + LangChain)
├── requirements.txt           # Dependências do Python
├── .env                       # Arquivo para suas chaves de API (não o compartilhe)
│
├── data/
│   ├── Scrum-Guide.pdf
│   ├── agile_principles.pdf
│   ├── agile-manifesto.pdf
│   └── data.pdf   
│
├── prompts/
│   └── scrum_master_prompt.json # Estrutura do prompt para o assistente
│
├── templates/
│   └── index.html             # Estrutura da página de chat
│
└── static/
    ├── css/
    │   └── style.css          # Estilos da página
    ├── js/
    │   └── script.js          # Lógica do frontend (interação do chat)
    └── img/
        └── copilot_avatar.png   # Imagem de avatar para o Copilot