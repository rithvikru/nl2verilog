#!/bin/bash
# Setup script for NL2Verilog dependencies

set -e

echo "NL2Verilog Tool Setup"
echo "===================="
echo

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
fi

echo "Detected OS: $OS"
echo

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create tools directory
TOOLS_DIR="$HOME/.nl2verilog_tools"
mkdir -p "$TOOLS_DIR"
echo "Installing tools to: $TOOLS_DIR"
echo

# Check and install Spot (for ltlsynt)
echo "Checking Spot..."
if command_exists ltlsynt; then
    echo "✓ Spot is already installed"
else
    echo "Installing Spot..."
    if [[ "$OS" == "macos" ]]; then
        if command_exists brew; then
            brew install spot
        else
            echo "Error: Homebrew not found. Please install it first."
            exit 1
        fi
    elif [[ "$OS" == "linux" ]]; then
        echo "Please install Spot using your package manager:"
        echo "  Ubuntu/Debian: sudo apt-get install spot"
        echo "  Or build from source: https://spot.lre.epita.fr/install.html"
    fi
fi

# Check and install syfco
echo
echo "Checking syfco..."
if command_exists syfco; then
    echo "✓ syfco is already installed"
else
    echo "Installing syfco..."
    cd "$TOOLS_DIR"
    if [[ ! -d "syfco" ]]; then
        git clone https://github.com/reactive-systems/syfco.git
    fi
    cd syfco
    echo "Please follow the syfco build instructions in the README"
    echo "Typically: stack install"
fi

# Check and install AIGER tools
echo
echo "Checking AIGER tools..."
if command_exists aigtoaig; then
    echo "✓ AIGER tools are already installed"
else
    echo "Installing AIGER tools..."
    cd "$TOOLS_DIR"
    if [[ ! -d "aiger" ]]; then
        git clone https://github.com/arminbiere/aiger.git
    fi
    cd aiger
    ./configure && make
    
    # Add to PATH or copy to /usr/local/bin
    if [[ -w /usr/local/bin ]]; then
        cp aigtoaig /usr/local/bin/
        echo "✓ Installed aigtoaig to /usr/local/bin"
    else
        echo "Please add $TOOLS_DIR/aiger to your PATH or copy aigtoaig to /usr/local/bin with sudo"
    fi
fi

# Check and install ABC
echo
echo "Checking ABC..."
if command_exists abc; then
    echo "✓ ABC is already installed"
else
    echo "Installing ABC..."
    cd "$TOOLS_DIR"
    if [[ ! -d "abc" ]]; then
        git clone https://github.com/berkeley-abc/abc.git
    fi
    cd abc
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
    
    # Add to PATH or copy to /usr/local/bin
    if [[ -w /usr/local/bin ]]; then
        cp abc /usr/local/bin/
        echo "✓ Installed abc to /usr/local/bin"
    else
        echo "Please add $TOOLS_DIR/abc to your PATH or copy abc to /usr/local/bin with sudo"
    fi
fi

# Python dependencies
echo
echo "Setting up Python environment..."
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    echo
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=your-api-key-here" > .env
    echo "Please edit .env and add your OpenAI API key"
fi

echo
echo "Setup complete!"
echo
echo "To use NL2Verilog:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run: python nl2verilog.py \"Your specification here\""
echo
echo "If any tools failed to install, please install them manually." 