## Prompt-1:

## Prompt-2:

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
