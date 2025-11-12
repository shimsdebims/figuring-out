#!/bin/bash

# CropGuard Model Deployment Script
# This script deploys the new MobileNetV2 model and UI improvements

set -e  # Exit on error

echo "================================================"
echo "🌱 CropGuard Model Deployment"
echo "================================================"
echo ""

# Navigate to project directory
cd /Users/shimasarah/Desktop/SHIMA/figuring-out

echo "📁 Current directory: $(pwd)"
echo ""

# Step 1: Backup existing files
echo "💾 Step 1: Backing up existing files..."
cp model.py model_backup.py 2>/dev/null || echo "   (model.py not found, skipping backup)"
cp requirements.txt requirements_backup.txt 2>/dev/null || echo "   (requirements.txt not found, skipping backup)"
echo "   ✅ Backups created (model_backup.py, requirements_backup.txt)"
echo ""

# Step 2: Replace files
echo "🔄 Step 2: Replacing with new files..."
if [ -f "model_v2.py" ]; then
    mv model_v2.py model.py
    echo "   ✅ model.py updated"
else
    echo "   ❌ ERROR: model_v2.py not found!"
    exit 1
fi

if [ -f "requirements_new.txt" ]; then
    mv requirements_new.txt requirements.txt
    echo "   ✅ requirements.txt updated"
else
    echo "   ❌ ERROR: requirements_new.txt not found!"
    exit 1
fi
echo ""

# Step 3: Check Git status
echo "📊 Step 3: Checking Git status..."
git status --short
echo ""

# Step 4: Stage changes
echo "📝 Step 4: Staging changes..."
git add model.py requirements.txt streamlit_app.py
echo "   ✅ Files staged for commit"
echo ""

# Step 5: Commit
echo "💬 Step 5: Committing changes..."
git commit -m "Implement MobileNetV2 model with UI improvements

- Replace TFLite model with MobileNetV2 from Hugging Face
- Add automatic 38→16 class mapping
- Implement confidence-based result display
- Add unmapped disease handling
- Add image quality tips for low confidence predictions
- Update model file check message"

echo "   ✅ Changes committed"
echo ""

# Step 6: Push to GitHub
echo "🚀 Step 6: Pushing to GitHub..."
git push origin main
echo "   ✅ Pushed to remote repository"
echo ""

echo "================================================"
echo "✅ Deployment Complete!"
echo "================================================"
echo ""
echo "🌐 Your app will redeploy automatically on Streamlit Cloud"
echo "🕐 Expected deployment time: 2-3 minutes"
echo "🔗 App URL: https://figuring-out-advdrzyhouwi2axrrwgstu.streamlit.app/"
echo ""
echo "📋 Next steps:"
echo "   1. Wait for Streamlit Cloud to finish redeploying"
echo "   2. Visit the app URL to test"
echo "   3. Check that model downloads successfully"
echo "   4. Test with example images from Assets/ folder"
echo ""
echo "📚 Documentation:"
echo "   - Full guide: IMPLEMENTATION_GUIDE.md"
echo "   - Web app changes: WEB_APP_UPDATES.md"
echo "   - Quick reference: QUICK_START.md"
echo ""
echo "🆘 If something goes wrong:"
echo "   - Restore backups: mv model_backup.py model.py"
echo "   - Check Streamlit logs in the dashboard"
echo "   - Run test_model.py locally to debug"
echo ""
echo "Good luck! 🍀"
