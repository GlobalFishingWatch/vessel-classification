from __future__ import print_function
import posixpath as pp
import subprocess
import sys
import tempfile

def fix_mmsi(path):
    command = ['gsutil', 'ls', pp.join(path, 'features')]
    print("Running:", ' '.join(command))
    file_list = subprocess.check_output(command, bufsize=-1).split()
    mmsi = [pp.basename(x).split('.')[0] for x in file_list]
    text = '\n'.join(mmsi)
    with tempfile.NamedTemporaryFile(delete=False) as output:
        output.write(text)
    command = ['gsutil', 'mv', output.name, pp.join(path, 'vessel_ids/part-00000-of-00001.txt')]
    print("Running:", ' '.join(command))
    subprocess.check_call(command)



if __name__ == "__main__":
    assert len(sys.argv) == 2
    path = sys.argv[1]
    print("Fixing ", path)
    fix_mmsi(path)
