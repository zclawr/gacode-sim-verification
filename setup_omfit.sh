#!/bin/bash

# Exit on error
set -e

# Variables
OMFIT_BRANCH="xarray-fix"
OMFIT_REPO_URL="git@github.com:wesleyliu728/OMFIT-source.git" # Replace with your fork
OMFIT_DIR="OMFIT-source"

# Step 3: Clone the OMFIT-source repository from the specific branch
if [ ! -d "$OMFIT_DIR" ]; then
    echo "🔽 Cloning OMFIT-source from branch '$OMFIT_BRANCH'..."
    git clone --branch "$OMFIT_BRANCH" "$OMFIT_REPO_URL" "$OMFIT_DIR"
else
    echo "📁 Directory '$OMFIT_DIR' already exists. Skipping clone."
fi

echo "✅ Setup complete. OMFIT now installed."