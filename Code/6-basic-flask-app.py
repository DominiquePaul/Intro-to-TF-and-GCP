# Just run this script in your terminal or in a code editor like sublime (do not use an interactive environment like Spyder)

from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
	return("Hello World")


if __name__ == "__main__":
	app.run(debug=True)