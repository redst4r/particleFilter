#!/bin/bash
rsync -av --progress --delete --numeric-ids /home/michi/pythonProjects/particleFilter/ michael.strasser@$HNODE_IP:$SERVER_HOME/pythonProjects/particleFilter
