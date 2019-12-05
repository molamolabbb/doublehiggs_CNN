import glob
import os, sys
import ROOT

def main():
  data_name = sys.argv[1:][0]
  path_name = '/home/jua/doublehiggs_jetimage/data/{}/{}.root'.format(data_name,data_name)
  split_dir = './split/{}'.format(data_name)
  if not os.path.isdir(split_dir):
    print "Make Folder: ", split_dir
    os.mkdir(split_dir)
  else: 
    print"Folder is already exist: ", split_dir
    sys.exit()
  f = ROOT.TFile(path_name)
  tree = f.events
  num_entries = float(tree.GetEntries())
  commands = [ 'rooteventselector -l {} {}:events {}/train.root'.format(int(num_entries*6./10.), path_name, split_dir),
              'rooteventselector -f {} {}:events {}/last.root'.format(int(num_entries*6./10.), path_name, split_dir),
              'rooteventselector -l {} {}/last.root:events {}/validation.root'.format(int(num_entries*2./10.), split_dir, split_dir),
              'rooteventselector -f {} {}/last.root:events {}/test.root'.format(int(num_entries*2./10.), split_dir, split_dir)
              ]
  for command in commands:
    os.system(command)
  #os.system('rm {}/last.root'.format(split_dir))
if __name__ == '__main__':
  main()
