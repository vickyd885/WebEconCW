if [ $1 = "" ]; then
  echo "You forgot to provide a file name!"
else
  cmd="curl http://deepmining.cs.ucl.ac.uk/api/upload/wining_criteria_1/YV1TonEheAPg  -X Post -F 'file=@"$1"'"
  echo "Calling:.."$cmd
  eval $cmd
fi
