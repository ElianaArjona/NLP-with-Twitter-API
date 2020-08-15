from heapq import nlargest
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import codecs
from gensim.summarization import summarize

document1 ="""Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems 
use to progressively improve their performance on a specific task. Machine learning algorithms build a mathematical model 
of sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to 
perform the task. Machine learning algorithms are used in the applications of email filtering, detection of network intruders, 
and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning 
is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization 
delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, 
and focuses on exploratory data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics."""

document2 = """Our Father who art in heaven, hallowed be thy name. Thy kingdom come. Thy will be done, on earth as it is 
in heaven. Give us this day our daily bread; and forgive us our trespasses, as we forgive those who trespass against us; 
and lead us not into temptation, but deliver us from evil
"""

document3 = """Hoy en día los trabajadores cuentan con mayor cantidad de herramientas para realizar sus trabajos, de la misma forma, la cantidad de trabajo que realizaban se duplica cada vez más debido al rápido aumentos de la cantidad de empresas y clientes que los avances tecnológicos han traído consigo. Hace 20 años, la oportunidades que tenía un empresa de surgir y dar a conocer su marca no son las misma que tenemos hoy en día, debido al gran auge que han causado la redes sociales, herramientas como Customer Relationship Management (CRM), Google Analytics, Yahoo Small Business, entre otros, son herramientas que brindan apoyo y soporte de una forma económica, fácil y accesible para cualquier empresa, sin importar su tamaño, ubicación, población de clientes, servicios ofrecidos entre otras cosas.
Gracias a los avances tecnológicos también podemos ver otro fenómeno muy popular en los últimos años, este es conocido como la digitalización, este consiste en convertir en formato digital todo aquello que alguna vez se realizó de forma manual como por ejemplo, llenar formularios, crear carpetas con historia de vida de personas, actividades, repartir información entre los miembros de un equipo, sociedad, comunidad, etc, inclusive, algo tan tradicional como la noticias del día a día, se están publicando de forma digital a través de páginas webs, redes sociales, entre otros. La digitalización es un gran paso que están dando muchas empresas hoy en día, y esto ayuda reducir la cantidad de papel empleado para estas labores, brinda mayor seguridad a la información, se manejan repasados de la misma y se tiene un mejor control y accesos a los datos siempre que sea necesario.
Los dos aspectos antes mencionado son puntos claves para dar un nuevo paso que ayudará no solo a reducir la cantidad de tareas que realiza un empleado en su día a día, también ayuda a reducir el esfuerzo, tiempo y dinero que se dedica para que estas tareas se cumplan, este nuevo cambio es conocido como Automatización Robótica de Procesos (RPA). La automatización de procesos a través de bots brinda gran ayuda a los empleados, esta automatización no es más que una simulación exacta del paso a paso que realiza una persona para ejecutar su tarea diaria, la única diferencia está en que al realizarla el bot y no el humano, se obtienen beneficios como aumento de la velocidad con que se ejecuta la tarea, reducción de errores humanos, reducción de tiempo y esfuerzo empleado por la persona en realizar esta tarea en específico, por lo que puede enfocarse en otras tareas de su trabajo, también, los bots puede ser ejecutar a cualquier hora, solo deben ser programadas las horas la que se necesita que se ejecute, esto ayuda a eliminar la diferencias horarias que puedan existir entre los países con los que colabora la empresa. 
Este trabajo de grado está enfocado en la implementación de Automatización Robótica de Procesos en una empresa, se explicará cuáles son las características que debe tener un proceso para poder ser automatizado, así como también, se detallará los beneficios que la implementación de RPA trae consigo. Para realizar la automatización de procesos, se utilizará la herramienta de Automation Anywhere en su versión Community Edition, esta trae consigo un gran repertorio de funcionalidades e integraciones de aplicaciones comúnmente utilizadas por la empresa en sus procesos diarios, entre estas Microsoft Office Excel, Sistemas, Aplicaciones y Productos para el procesamiento de datos (SAP) y archivos formato PDF, esta herramienta permite que la adopción de RPA en cualquier ambiente laboral sea un transición fácil y agradable para todos sus empleados, sin retrasar o afectar en ningún punto las tareas cotidianas de los usuarios. 
"""
def removeSpecialChar(sentence):

    remove_alfa = ['ñ', 'á', 'é', 'í', 'ó', 'ú', 'ü']

    for i in range(len(remove_alfa)):
        if i == 0:
            sentence = re.sub(remove_alfa[i], 'n', sentence)
        elif i == 1:
            sentence = re.sub(remove_alfa[i], 'a', sentence)
        elif i == 2:
            sentence = re.sub(remove_alfa[i], 'e', sentence)
        elif i == 3:
            sentence = re.sub(remove_alfa[i], 'i', sentence)
        elif i == 4:
            sentence = re.sub(remove_alfa[i], 'o', sentence)
        elif i == 5 or i == 6:
            sentence = re.sub(remove_alfa[i], 'u', sentence)

    return sentence

"""
document = []
count = 0
f = open("./summary/test.txt", "r", encoding="utf-8")

for x in f:
    print(removeSpecialChar(x))
    if count %2 == 0:
      document.append(removeSpecialChar(x))
    count+=1
f.close()

document = " ".join(document)"""


#nlp = spacy.load(r'C:\Users\eliana\AppData\Local\Programs\Python\Python38\Lib\site-packages\en_core_web_sm\en_core_web_sm-2.3.0')
nlp = spacy.load('./es_core_news_lg')
stopwords = list(STOP_WORDS)

def text_summarizer(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)

    # Build Word Frequency
# word.text is tokenization in spacy
    word_frequencies = {}
    for word in docx:
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Sentence Tokens
    sentence_list = [ sentence for sentence in docx.sents ]

    # Calculate Sentence Score and Ranking
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]


    # Find N Largest
    summary_sentences = nlargest(4, sentence_scores, key=sentence_scores.get)
    final_sentences = [ w.text for w in summary_sentences ]

    createTxt(final_sentences)

    summary = ' '.join(final_sentences)


    print("Original Document\n")
    # print(raw_docx)
    print("Total Length:",len(raw_docx))
    print('\n\nSummarized Document\n')
    # print(summary)
    print("Total Length:",len(summary))

    f = open("./summary/summary-yt.txt", "w+", encoding="utf-8")
    f.write(raw_docx)
    f.close()

    f = open("./summary/original-yt.txt", "w+", encoding="utf-8")
    f.write(summary)
    f.close()


def readingTime(docs):
    total_words_tokens =  [ token.text for token in nlp(docs)]
    estimatedtime  = len(total_words_tokens)/200
    return '{} mins'.format(round(estimatedtime))

def createTxt(txt):
    f = codecs.open('./summary/result.txt',"w+","utf-8")
    for i in range(len(txt)):
        f.write(txt[i])
    f.close()

text_summarizer(document1)


# Compare with Gensim version

gn = summarize(document3)

print("gensim :", gn)