import aiml
import os

# Create the kernel and learn AIML files
kernel = aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("load aiml b")

# Press CTRL-C to break this loop
while True:
    rep = raw_input("Enter your message >> ")
    if rep == "quit":
        exit()    
    else:
        print(kernel.respond(rep))
