 
from openai import OpenAI
import time 
from pathlib import Path

def sendPrompt(prompt, file, openai_api_key ):
#steps from https://platform.openai.com/docs/assistants/overview
  client = OpenAI(
    api_key=openai_api_key,
  )

  assistant = client.beta.assistants.create(
    name="Cat Detector", # You need to specify your app name here
    description="You detect if a frame has a cat only return 'TRUE' OR 'FALSE'. nothing else  ", # Your assistant’s description
    model="gpt-4-1106-preview", # The model you want to use
    tools=[{"type": "code_interpreter"}]
    
  )

  #to pass a file
  file = client.files.create(
        file=Path("frame.jpeg") 
      , purpose='assistants'
      )

  thread = client.beta.threads.create()

  message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role="user",
      content=prompt, #"Is there a cat in the picture ? say TRUE OR FALSE" + str(plot) 
      file_ids=[file.id]
  )

  run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
  )
  
  completed = "FALSE" 

  while completed == "FALSE" :
    run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
      run_id=run.id
    )
    print ("run statue is : " + str(run.status))
    time.sleep(5)
    if run.status == 'completed':
      messages = client.beta.threads.messages.list(
        thread_id=thread.id
      )
      completed == "TRUE"
      for message in messages.data:          
        for content in message.content:  # Assuming message.content is a list
          if content.type == 'text':  # Check if the content type is 'text'
            return content.text.value  # Access the 'value' attribute of the 'text' field
         