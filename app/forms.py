from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, IntegerField
from wtforms.validators import DataRequired

class SubmitTextForm(FlaskForm):
    text = TextAreaField('Text to Summarize', validators=[DataRequired()])
    num_sentences = IntegerField('Number of Sentences', validators=[DataRequired()], default=3)
    submit = SubmitField('Summarize')