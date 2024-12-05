import csv
from openai import OpenAI

save_file = "dataset-big.csv"

# Get API key
with open("api.txt") as api:
    client = OpenAI(api_key=api.read().strip())  # Strip any extra whitespace

# Get initial instructions
with open("prompt.txt") as prompt_file:
    assistant = client.beta.assistants.create(
        name="sumaryzer1",
        instructions=prompt_file.read(),
        model="gpt-4o-mini",
    )

# Start a message thread with the assistant
thread = client.beta.threads.create()

# Load prompts from file
prompts = []
with open('promptlist-short.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        prompts.append(row[0])

# Add prompts to the thread and collect responses
for prompt in prompts[:250]:
    # Send the user prompt to the assistant
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )

    # Run the assistant
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Check if the run completed successfully
    if run.status == 'completed':
        # Retrieve messages from the thread
        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            limit=limit,
        )

        # Get the raw log and summarised log
        raw_logs = None
        summarised_logs = None

        # Extract relevant information from messages
        for message in messages.data:
            if message.role == "user":
                raw_logs = message.content
            elif message.role == "assistant":
                summarised_logs = message.content

        # Make sure both raw and summarised logs are not None
        if raw_logs and summarised_logs:
            # Append the logs to the CSV file
            with open(save_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([raw_logs, summarised_logs])

    else:
        print(f"Run status: {run.status}")
