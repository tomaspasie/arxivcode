#!/bin/bash
# Setup environment variables from .env file

if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo ""
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "Please edit .env and add your API keys:"
    echo "  - GITHUB_TOKEN (for data collection)"
    echo "  - OPENAI_API_KEY (for LLM explanations)"
    echo ""
    echo "Get your keys at:"
    echo "  - GitHub: https://github.com/settings/tokens"
    echo "  - OpenAI: https://platform.openai.com/api-keys"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

echo "✅ Environment variables loaded from .env"
echo ""
echo "Loaded variables:"
if [ -n "$GITHUB_TOKEN" ]; then
    echo "  ✅ GITHUB_TOKEN: ${GITHUB_TOKEN:0:10}..."
else
    echo "  ❌ GITHUB_TOKEN: not set"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo "  ✅ OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."
else
    echo "  ❌ OPENAI_API_KEY: not set"
fi
echo ""
