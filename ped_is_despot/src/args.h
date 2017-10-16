#include <libplayerc++/playerc++.h>
#include <iostream>
#include <unistd.h>

std::string  gHostname(PlayerCc::PLAYER_HOSTNAME);
uint         gPort(PlayerCc::PLAYER_PORTNUM);
uint         gIndex(0);
uint         gDebug(0);
uint         gFrequency(10); // Hz
uint         gDataMode(PLAYER_DATAMODE_PUSH);
bool         gUseLaser(false);

void print_usage(int argc, char** argv);

/// MoveTargetPos moves the robot and returns the success or failure 
/// of reading the target.path file
int MoveTargetPos();

int parse_args(int argc, char** argv)
{
  // set the flags
  const char* optflags = "h:p:i:d:u:lm:";
  int ch;

  // use getopt to parse the flags
  while(-1 != (ch = getopt(argc, argv, optflags)))
  {
    switch(ch)
    {
      // case values must match long_options
      case 'h': // hostname
          gHostname = optarg;
          break;
      case 'p': // port
          gPort = atoi(optarg);
          break;
      case 'i': // index
          gIndex = atoi(optarg);
          break;
      case 'd': // debug
          gDebug = atoi(optarg);
          break;
      case 'u': // update rate
          gFrequency = atoi(optarg);
          break;
      case 'm': // datamode
          gDataMode = atoi(optarg);
          break;
      case 'l': // datamode
          gUseLaser = true;
          break;
      case '?': // help
      case ':':
      default:  // unknown
        print_usage(argc, argv);
        exit (-1);
    }
  }

  return (0);
} // end parse_args

void print_usage(int argc, char** argv)
{
  using namespace std;
  cerr << "USAGE:  " << *argv << " [options]" << endl << endl;
  cerr << "Where [options] can be:" << endl;
  cerr << "  -h <hostname>  : hostname to connect to (default: "
       << PlayerCc::PLAYER_HOSTNAME << ")" << endl;
  cerr << "  -p <port>      : port where Player will listen (default: "
       << PlayerCc::PLAYER_PORTNUM << ")" << endl;
  cerr << "  -i <index>     : device index (default: 0)"
       << endl;
  cerr << "  -d <level>     : debug message level (0 = none -- 9 = all)"
       << endl;
  cerr << "  -u <rate>      : set server update rate to <rate> in Hz"
       << endl;
  cerr << "  -l      : Use laser if applicable"
       << endl;
  cerr << "  -m <datamode>  : set server data delivery mode"
       << endl;
  cerr << "                      PLAYER_DATAMODE_PUSH = "
       << PLAYER_DATAMODE_PUSH << endl;
  cerr << "                      PLAYER_DATAMODE_PULL = "
       << PLAYER_DATAMODE_PULL << endl;
/*  cerr << "                      PLAYER_DATAMODE_PUSH_ALL = "
       << PLAYER_DATAMODE_PUSH_ALL << endl;
  cerr << "                      PLAYER_DATAMODE_PULL_ALL = "
       << PLAYER_DATAMODE_PULL_ALL << endl;
  cerr << "                      PLAYER_DATAMODE_PUSH_NEW = "
       << PLAYER_DATAMODE_PUSH_NEW << endl;
  cerr << "                      PLAYER_DATAMODE_PULL_NEW = "
       << PLAYER_DATAMODE_PULL_NEW << endl;
  cerr << "                      PLAYER_DATAMODE_ASYNC    = "
       << PLAYER_DATAMODE_ASYNC << endl;*/
} // end print_usage




void print_usage(const char* cmdName) 
{
	cout << "Usage: " << cmdName << " POMDPModelFileName --policy-file policyFileName --simLen numberSteps \n" 
<<"	--simNum numberSimulations [--fast] [--srand randomSeed] [--output-file outputFileName]\n" 
<<"    or " << cmdName << " --help (or -h)  Print this help\n" 
<<"    or " << cmdName << " --version	  Print version information\n" 
<<"\n"
<<"Simulator options:\n"
<<"  --policy-file policyFileName	Use policyFileName as the policy file name (compulsory).\n"
<<"  --simLen numberSteps		Use numberSteps as the number of steps for each\n" 
<<"				simulation run (compulsory).\n"
<<"  --simNum numberSimulations	Use numberSimulations as the number of simulation runs\n" 
<<"				(compulsory).\n"
<<"  -f or --fast			Use fast (but very picky) alternate parser for .pomdp files.\n"
<<"  --srand randomSeed		Set randomSeed as the random seed for simulation.\n" 
<<"				It is the current time by default.\n"
//<<"  --lookahead yes/no		Set 'yes' ('no') to select action with (without) one-step\n" 
//<<"				look ahead. Action selection is with one-step look ahead\n" 
//<<"				by default.\n" 
<<"\n"
<<"Output options:\n"
<<"  --output-file outputFileName	Use outputFileName as the name for the output file\n" 
<<"				that contains the simulation trace.\n"
		<< "Example:\n"
		<< "  " << cmdName << " --simLen 100 --simNum 100 --policy-file out.policy Hallway.pomdp\n";

// 	cout << "usage: binary [options] problem:\n"
// 		<< "--help, print this message\n"
// 		<< "--policy-file, policy file to be used\n"
// 		<< "--output-file, output file to be used\n"
// 		<< "--simLen, length of simulation\n"
// 		<< "--simNum, number of simulations\n"
// 		<< "--srand, random seed (default: current time)\n"
// 		<< "--lookahead, use \"one-step look ahead\" when selecting action (default: yes)\n"
// 		<< "Examples:\n"
// 		<< " ./simulate --simLen 100 --simNum 100 --policy-file out.policy Hallway.pomdp\n";

}

void generateSimLog(SolverParams& p, double& globalExpRew, double& confInterval)
{
    int length;
    char str1[102];
    string str_comb;

    int startpos = 0;
    int i;
    for (i = p.problemName.length() - 1; i >= 0; i--) {
        if (p.problemName[i] == '/') {
            startpos = i + 1;
            break;
        }
    }

    str_comb.append(p.problemName.begin() + startpos, p.problemName.end());

    str_comb.append("SimLog");
    cout << str_comb << endl;

    length = str_comb.copy(str1, 100);
    str1[length] = '\0';

    FILE *fp = fopen(str1, "a");

    //  FILE *fp = fopen("sim.log","a");
    if (fp == NULL) 
    {
        cerr << "cant open sim logfile\n";
        exit(1);
    }

    fprintf(fp, "%f ", globalExpRew);
    fprintf(fp, "%f ", globalExpRew - confInterval);
    fprintf(fp, "%f ", globalExpRew + confInterval);
    fprintf(fp, "\n");
    fclose(fp);


}



