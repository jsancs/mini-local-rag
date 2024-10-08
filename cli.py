from prompt_toolkit.shortcuts import prompt


def main():
    while True:
        try:
            answer = prompt(
                "Give me some input: ",
                placeholder='Send a message (/? for help)',
            )
            print('You said: %s' % answer)
        except KeyboardInterrupt:
            # Ctrl-C interrupt
            break
        except EOFError:
            # Ctrl-D to exit
            break

if __name__ == '__main__':
    main()