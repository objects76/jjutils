
import argparse

def main(args):
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '-agument', help='Exaple argumet', required=True)
    args = parser.parse_args()
    main(args)