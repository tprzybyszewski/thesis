from django import forms

class LogEntryForm(forms.Form):
    original_log = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 10, 'cols': 50}),
        label="Original Log"
    )
    summarized_log = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 10, 'cols': 50, 'readonly': True}),
        required=False,
        label="Summarized Log"
    )
    question = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 2, 'cols': 50}),
        required=False,
        label="Your Question"
    )
    answer = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 2, 'cols': 50, 'readonly': True}),
        required=False,
        label="Answer from AI"
    )
