from flask import render_template, request
from app import app
from app.forms import SubmitTextForm
from app.summarizer import Summarizer, nlp
from app.translation import translate_text
from werkzeug.utils import secure_filename
import os
from googletrans import Translator

ALLOWED_EXTENSIONS = {'txt', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    summarizer = Summarizer(nlp)
    form = SubmitTextForm(size=600)

    if request.method == 'POST':
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
                file_text = uploaded_file.read().decode('utf-8')
                
                num_sentences = int(request.form['num_sentences'])
                word_weights, sentence_weights, sents, summary = summarizer.summarize_text(file_text, num_sentences)                
                top_five_words = sorted(word_weights, key=word_weights.get, reverse=True)[:5]
                sentence_weights = [value for key, value in sentence_weights.items()]
                weighted_sentence_weights = [value/max(sentence_weights) for value in sentence_weights]
                sentences_with_weights = list(zip(sents, weighted_sentence_weights))
                target_language = request.form['target_language']  
                translated_summary = translate_text(summary, target_language)
                return render_template('summary.html',  text=translated_summary, top_words=top_five_words, sentence_weights=sentence_weights, sents=sentences_with_weights)

        elif form.validate_on_submit():
            text = request.form['text']
            num_sentences = int(request.form['num_sentences'])
              # Detect language
            translator = Translator()
            detected_lang = translator.detect(text).lang
            
            # Translate to English if not in English
            if detected_lang != 'en':
                translated_text = translator.translate(text, src=detected_lang, dest='en').text
            else:
                translated_text = text
            word_weights, sentence_weights, sents, summary = summarizer.summarize_text(translated_text, num_sentences)
            top_five_words = sorted(word_weights, key=word_weights.get, reverse=True)[:5]
            sentence_weights = [value for key, value in sentence_weights.items()]
            weighted_sentence_weights = [value/max(sentence_weights) for value in sentence_weights]
            sentences_with_weights = list(zip(sents, weighted_sentence_weights))
            target_language = request.form['target_language']  
            translated_summary = translate_text(summary, target_language)
            return render_template('summary.html',  text=translated_summary, top_words=top_five_words, sentence_weights=sentence_weights, sents=sentences_with_weights)

    return render_template('index.html', text='', form=form)

@app.route('/summary')
def summary():
    return render_template('summary.html')
