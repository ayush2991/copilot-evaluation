import pandas as pd

# A list of 10 diverse queries sent to ChatGPT model
queries = [
    "What is the capital of France?",
    "Can you explain the theory of relativity?",
    "What are the main differences between Python and Java?",
    "How do you make a perfect cup of coffee?",
    "What are the health benefits of green tea?",
    "Can you provide a summary of the book '1984' by George Orwell?",
    "What are the latest trends in artificial intelligence?",
    "How do you start a small business?",
    "What are the best practices for maintaining a healthy diet?",
    "Can you give me a brief overview of the history of the internet?"
]

# A list of responses from the ChatGPT model to the above queries
responses = [
    "The capital of France is Paris.",
    "The theory of relativity, proposed by Albert Einstein, describes the relationship between space and time. It consists of two theories: special relativity and general relativity.",
    "Python is an interpreted, high-level programming language known for its readability and simplicity, while Java is a compiled, object-oriented language that is platform-independent due to the Java Virtual Machine (JVM).",
    "To make a perfect cup of coffee, use freshly ground coffee beans, filtered water at the right temperature (195-205°F), and brew for about 4-5 minutes.",
    "Green tea is rich in antioxidants, may improve brain function, aids in fat loss, and has been linked to a lower risk of heart disease.",
    "'1984' is a dystopian novel that explores themes of totalitarianism, surveillance, and individuality in a society controlled by the Party and its leader, Big Brother.",
    "Latest trends in AI include advancements in natural language processing, machine learning algorithms, ethical AI practices, and AI applications in various industries.",
    "To start a small business, identify a market need, create a business plan, secure funding, register your business, and develop a marketing strategy.",
    "Best practices for maintaining a healthy diet include eating a variety of foods, focusing on whole grains, fruits, vegetables, lean proteins, and healthy fats while limiting processed foods and sugars.",
    "The history of the internet began in the 1960s with ARPANET. It evolved through various stages including the development of TCP/IP protocols in the 1970s and the World Wide Web in the 1990s."
]

# Create a dataframe to store the queries and responses
df = pd.DataFrame({
    'Query': queries,
    'Response': responses
})

# Preview the dataframe
print(df.head())

# Perform NER on the query and response columns
import spacy
nlp = spacy.load('en_core_web_sm')
for col in ['Query', 'Response']:
    df[col + '_NER'] = df[col].apply(lambda x: [(ent.text, ent.label_) for ent in nlp(x).ents])

# Display the dataframe with NER results
print(df[['Query', 'Query_NER', 'Response', 'Response_NER']].head())