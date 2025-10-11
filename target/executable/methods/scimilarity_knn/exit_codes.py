import sys

# when the method is not applicable to the input data,
# exit with code 99
def exit_non_applicable(msg):
    print(f"NON-APPLICABLE ERROR: {msg}", flush=True)
    sys.exit(99)