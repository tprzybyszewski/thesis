import os
from mistralai import Mistral
import csv
import time
import re


client = Mistral(api_key="fr")
save_file = "dataset-metrics-fin.csv"


def remove_hashes_and_asterisks(text):
    cleaned_text = text.replace('###', '').replace('**', '')
    return cleaned_text

def clean_ticket(ticket_text):
    cleaned_text = re.sub(r"^-{2,}\s", "- ", ticket_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r"-\s+", "- ", cleaned_text)
    cleaned_text = cleaned_text.strip('"')
    return cleaned_text

def clean_bullet_points(text):
    """
    This function removes double dashes and excess whitespace 
    from bullet points in the given text, ensuring each line
    starts with a single dash.
    """
    lines = text.splitlines()
    
    cleaned_lines = []
    for line in lines:
        stripped_line = line.lstrip()
        
        if stripped_line.startswith("--"):
            cleaned_line = "- " + stripped_line[2:].lstrip()  
        elif stripped_line.startswith("- -"):
            cleaned_line = "- " + stripped_line[3:].lstrip()  
        elif stripped_line.startswith("-"):
            cleaned_line = "- " + stripped_line[1:].lstrip()  
        else:
            cleaned_line = line 
        cleaned_lines.append(cleaned_line)
    
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text

prompts = []
with open('promptlist-short.csv', mode='r') as file:
  csv_reader = csv.reader(file)
  for row in csv_reader:
    prompts.append(row[0])


for prompt in prompts[100:]:
    chat_response = client.agents.complete(
        agent_id="ag:3cf886ba:20241007:eng-agent:cdc6aee3",
        messages=[
            {
                "role": "user",
                "content": str(prompt),
            },
        ],
    )


    raw_logs = prompt
    summarised_logs = clean_bullet_points(clean_ticket(chat_response.choices[0].message.content))

    print(raw_logs)
    print(summarised_logs)
    summarised_logs = remove_hashes_and_asterisks(summarised_logs)  
    with open(save_file, mode='a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([raw_logs, summarised_logs])
    time.sleep(1)

