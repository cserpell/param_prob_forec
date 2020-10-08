# coding=utf-8
"""Script to fix GEFCom2014 dates."""
import argparse
import datetime


def parse_file(file_name):
    """Perform dates fix."""
    started = False
    current_time = datetime.datetime.strptime('2001-01-01-00:00:00',
                                              '%Y-%m-%d-%H:%M:%S')
    step = datetime.timedelta(hours=1)
    with open(file_name) as file:
        while True:
            line = file.readline()
            if not line:
                break
            if not started:
                started = True
                print(line, end='')
                continue
            parts = line.split(',')
            current_time += step
            print(*([current_time.strftime('%Y-%m-%d %H:%M:%S')] +
                    parts[2:]), sep=',', end='')


def main():
    """Main execution point."""
    parser = argparse.ArgumentParser(description='Fix dates of GEFCom2014.')
    parser.add_argument(
        '--file_name', help='file name with input data',
        default='~/datasets/input.csv')
    args = parser.parse_args()
    parse_file(args.file_name)


if __name__ == '__main__':
    main()
