from app import app

if __name__ == '__main__':
    print("Starting server at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True) 