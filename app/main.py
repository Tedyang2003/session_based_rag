from flask import Flask
from routes.rag_routes_vlm import rag_bp


def create_app():
    app = Flask(__name__)

    app.register_blueprint(rag_bp, url_prefix='/rag')

    return app

if __name__=='__main__':
    app = create_app()
    app.run()