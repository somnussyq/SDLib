from averageAttack import AverageAttack
from bandwagonAttack import BandWagonAttack
from randomAttack import RandomAttack
from RR_Attack import RR_Attack
from hybridAttack import HybridAttack

attack = RR_Attack('./config/config.conf')
attack.insertSpam()
attack.farmLink()
attack.generateLabels('flabels.txt')
attack.generateProfiles('fprofiles.txt')
attack.generateSocialConnections('frelations.txt')