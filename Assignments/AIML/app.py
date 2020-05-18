import aiml
import os

kernel = aiml.Kernel()
if os.path.isfile("bot_brain.brn"):
    kernel.bootstrap(brainFile = "bot_brain.brn")

else:
    kernel.bootstrap(learnFiles = "std-startup.xml", commands="load aiml b")
    kernel.saveBrain("bot_brain.brn")

while True:
    message = input("Human: ")    
    if message == "quit" or "exit":
        exit()
    elif message == "save":
        kernel.saveBrain("bot_brain.brn")
    else:
        print("AI: " + kernel.respond(message))
