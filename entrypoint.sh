#!/bin/bash
set -e

# Attente DB si nécessaire (si tu la gères ailleurs, commente ça)
# ./wait-for-it.sh $ECHOFORGE_DATABASE_HOST:$ECHOFORGE_DATABASE_PORT -t 30
echo "✅ Démarrage du service..."

echo "✅ Base prête"
echo "⚙️  Initialisation de la base si nécessaire..."
python echoforge/db/init_db.py

# Lance le script principal (ajuste selon ton app)
# Gradio écoute par défaut sur 7860 — Cloud Run attend 8080 ! Il faut forcer Gradio à écouter sur 8080
exec python chat_prototype.py