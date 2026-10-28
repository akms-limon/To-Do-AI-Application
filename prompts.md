# Four meaningful prompts given bellow where I explained failures encountered, and how I corrected them.

## Prompt-1: Initial prompt
### prompt:
I want to develop a AI task manager website for my internship. I have 2 days. I want to build it by learning, so don't give me full project code at once. First check my requirements and ask me if anything need clarification. Then give me the full architecture and explain how a user message will go through AI, validation and database to complete a task. After that make a step by step development plan, then we will start the project one step at a time. I'm using Linux, VS Code, Git. Keep the project lightweight and install dependency only when we actually need it. I have attached the project requirements as a source. Read the full source and then ask me the clarification. Next we will plan for your project step by step.

## Prompt-2: Improve search result relevance
### Prompt:
My semantic search is returning unrelated tasks also. When I ask Show unfinished tasks related to football, its showing demo and other tasks which have no relation with football. I want it to understand the meaning and return only relevant tasks with the requested filters. If there is no relevant task, it should say no matching tasks found. Dont return a task just because it is the nearest result in ChromaDB. First explain why this happening, then fix it without adding unnecessary dependencies or extra AI calls.
### Failure encountered:
Searching for unfinished football tasks returned “Finish the demo tomorrow” and “long task”. The search returned the nearest available embeddings without checking whether those results were relevant enough.
### Correction implemented:
Added relevance filtering using semantic distance and word matching in task titles and descriptions. Status filters still apply, and unrelated results can now produce a “No relevant tasks” response.

## Prompt-3: Improve task selection without requiring IDs

### Prompt:
I want users to update tasks naturally by referring to their titles
without having to provide an ID every time. For example, if there is one
task called “Play football,” saying “Change the football tasks priority
to low” should update it directly when the match is clear. If two or more
tasks could reasonably match, the assistant should explain the options and
ask which one the user means before changing anything. If no task matches,
it should ask for a clearer title instead of guessing. Keep the existing
validation rules, and allow completed tasks to be edited too.

### Failure encountered:
During testing, I asked to change the football tasks priority to low.
Although “Play football” was the only relevant task, the assistant still
required its numeric ID. The resolver handled exact titles directly but 
asked for confirmation for every semantic match, making natural-language
updates unnecessarily repetitive.

### Correction planned:
I will adjust target resolution so a unique, sufficiently clear title or semantic
match can be selected automatically. Ambiguous or weak matches will still require
clarification. I wont treat the highest-ranked search result as automatically
correct, because even an unrelated task can rank first.



## Prompt-4: Handle clarification based on the reply of a user or Context resoning

### Prompt:
Ai assistant jokhon akta specific task diye jigges korbe - ‘Is this the task you mean?’
Tokhon user ‘yes’ bolle oi task-ta select korbe ebong appropriate action nibe. Like
Context bujhe kaj korbe, all time jeno ID na dite hoy. Jemon keu jodi bole amar presantation
er task ta modify kore Priority high kore dao. Se presantation er task er priority high kore
dibe kono title ba ID jigges korbe na. Ekadhik candidate thakle shudhu ‘yes’ diye select
korbe na. konta bojhachhe seta clear korbe. User priority ba description diye task identify
korte chaileo ager conversation-er context dhore request-ta resolve korbe. Multiple task
delete howar possibility thakle affected task-gulo dekhaye confirmation nibe. Guess kore
kono task change ba delete korbe na.

### Failure encountered:
When the assistant asked me to confirm a task, it didn’t accept “yes.” Instead, it repeatedly asked for the task’s title or ID.

### Planned correction:
Context-aware confirmation, filter-based target resolution, and ambiguous operation confirmation flow.
