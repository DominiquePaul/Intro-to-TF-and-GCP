from google.cloud import language
from google.cloud.language import types
from google.cloud.language import enums
import os

# You have to change this to your own credentials
# You can sign up for Google Cloud Platform for free and receive 300$ in
# credit with which you can play around:
# https://cloud.google.com/
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/dominiquepaul/xJob/4-Lecture/Credentials/siaw-spark-91597d721c12.json"


### Writing a Function that analses a text for us ### 

def language_analysis(text):
    # invokes the instance through which we can communicate with Google's server
    client = language.LanguageServiceClient()
    # As an input to the API we need to format our input text as a special document type.
    # This document contains further information which tells the API how to read it.
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Run the sentiment analysis. This tells us what overall emotion of the
    # text is (good or bad) and how strong it is (0 to infinity)
    sent_analysis = client.analyze_sentiment(document)
    sentiment = sent_analysis.document_sentiment

    # When analysing for entities, the API will look at the specific words in
    # the text and try to identify their meaning in the text
    ent_analysis = client.analyze_entities(document)
    entities = ent_analysis.entities

    return(sentiment, entities)



### Example 1 ###
# We create a sample text and run it through the function to examine the performance
sample1 = "Angela Merkel's party, the CDU, performed very poorly in this years german elections"
sentiment, entities = language_analysis(sample1)

# print the results
print(sentiment.score, sentiment.magnitude)
print(entities)



### Example 2 ###
# One use case for the API might be a big company which wants to know what
# its customers are saying about their products online, but dont have the
# resources or time to do this by hand. For this they could use the API
# to analyse all of the products and be notified if something might be wrong
# with some of the products.
# source: https://www.galaxus.ch/de/s1/product/sony-playstation-classic-spielkonsole-9676457
sample2 = """Schlechteste Emulation der Spiele bisher, keine gute Anpassung auf HD-Fernseher. FINGER WEG.
Es fehlen ein paar wichtige Spiele der PS-Ära wie z.B. Gran Turismo, Tony Hawk oder Tomb Raider.
Die Spiele laufen praktisch alle schlechter als damals auf der PlayStation. GTA ist praktisch unspielbar und ruckelt mit unter 20 fps. Die mit Abstand schlechteste Emulation, die Sony je geliefert hat. Sämtliche Konsolen, von PS2 bis Vita liefern eine bessere Emulation der alten Spiele. Holt euch die gewünschten Games also lieber auf einer anderen von Sonys Plattformen, da laufen die besser. Das hier ist nur überteuerter Plastikmüll, der schnell wieder in der Ecke landet.
"""
sentiment, entities = language_analysis(sample2)

# print the results:
print(sentiment.score, sentiment.magnitude)
print(entities)
