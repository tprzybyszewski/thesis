from django.shortcuts import render, redirect
from .models import ConversationSession
from .forms import LogEntryForm
from mistralai import Mistral
import os


client = Mistral(api_key="")

def summarize_log(log_text):
    response = client.agents.complete(
        agent_id="ag:3cf886ba:20241007:eng-agent:cdc6aee3",
        messages=[
            {
                "role": "user",
                "content": f"{log_text}:",
            },
        ],
    )
    summary = response.choices[0].message.content
    return summary

def ask_question(original_log, summarized_log, question, conversation_history):
    messages = []
    if conversation_history:
        history_entries = conversation_history.strip().split('\n')
        for entry in history_entries:
            if entry.startswith("Q: "):
                messages.append({"role": "user", "content": entry[3:].strip()})
            elif entry.startswith("A: "):
                messages.append({"role": "assistant", "content": entry[3:].strip()})

    messages.append({"role": "user", "content": f"{original_log}"})
    if summarized_log:
        messages.append({"role": "assistant", "content": f"{summarized_log}"})

    messages.append({"role": "user", "content": f"{question}"})


    response = client.agents.complete(
        agent_id="ag:3cf886ba:20241013:untitled-agent:35b91216",
        messages=messages,
    )

    answer = response.choices[0].message.content if response.choices else "Brak odpowiedzi."
    return answer

def log_view(request):
    conversation_id = request.session.get('conversation_id')
    if not conversation_id:
        conversation = None
    else:
        try:
            conversation = ConversationSession.objects.get(id=conversation_id)
        except ConversationSession.DoesNotExist:
            conversation = None

    if request.method == "POST":
        summarize_form = LogEntryForm(request.POST, prefix='summarize')
        question_form = LogEntryForm(request.POST, prefix='question')

        if 'summarize' in request.POST:
            if summarize_form.is_valid():
                original_log = summarize_form.cleaned_data.get('original_log')
                summarized_log = summarize_log(original_log)

                conversation = ConversationSession.objects.create(
                    original_log=original_log,
                    summarized_log=summarized_log,
                    conversation_history=""
                )
                request.session['conversation_id'] = conversation.id

                summarize_form = LogEntryForm(initial={
                    'original_log': original_log,
                    'summarized_log': summarized_log
                }, prefix='summarize')

                question_form = LogEntryForm(initial={
                    'question': '',
                    'answer': ''
                }, prefix='question') 

        elif 'question-ask_question' in request.POST:
            if conversation:
                question = request.POST.get('question-question')
                if question:
                    answer = ask_question(conversation.original_log, conversation.summarized_log, question, conversation.conversation_history)
                    conversation.conversation_history += f"\nQ: {question}\nA: {answer}\n"
                    conversation.save()

                    summarize_form = LogEntryForm(initial={
                        'original_log': conversation.original_log,
                        'summarized_log': conversation.summarized_log
                    }, prefix='summarize')

                    question_form = LogEntryForm(initial={
                        'question': question,
                        'answer': answer 
                    }, prefix='question')
                else:
                    question_form.add_error('question', 'Pytanie nie może być puste.')

    else:
        summarize_form = LogEntryForm(prefix='summarize')
        question_form = LogEntryForm(prefix='question')

    conversation_sessions = ConversationSession.objects.all().order_by('-created_at')
    print(conversation_sessions[0].original_log)

    return render(request, 'logs/log_view.html', {
        'summarize_form': summarize_form,
        'question_form': question_form,
        'conversation': conversation,
        'conversation_sessions': conversation_sessions
    })