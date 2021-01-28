import sys

def progress_bar(curr_iteration, total_iterations, num_bars=30, output_vals={}):
    """
        Shows a progress bar when called inside a loop
        params:
            curr_iteration, total_iterations : self_exp
            num_bars
            output_vals: dictionary of values to be displayed
    """
    static prevlen = 0
    if(curr_iteration==0):
        print(output_vals)
    total_iter = total_iterations
    total_iterations -= total_iterations%num_bars
    percent = min(100*curr_iteration/total_iterations, 100.00)
    bars = int((num_bars*percent)/100)
    done = '[' + bars*'='
    todo = (num_bars - bars)*'~' + ']'
    valstring = ""
    for i, val in enumerate(output_vals):
        valstring += val + " : " + f"{(output_vals[val]):.2f}" + " "
    print(f"\r{percent} {done+todo} {valstring}", end = "")
    sys.stdout.flush()

    if(curr_iteration == total_iter-1): 
        sys.stdout.flush()
    prevlen +=1 

    

if __name__ == "__main__":
    cost  = 3000
    acc = 0.9
    for i in range(1000):
        progress_bar(i, 1000, num_bars = 30, output_vals={"Cost":cost, "Accuracy":acc})
        cost/=3
        acc *= 1.001
    print('\n', end = "")
