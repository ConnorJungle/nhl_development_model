select 
	playerid,
	player,
	position,
	league,
	year,
	array_agg(team) as team,
	array_agg(teamid) as teamids,
	sum(gp) as gp, 
	sum(g) as g,
	sum(a) as a,
	sum(tp) as tp,
	sum(g) / sum(gp) as gpg,
	sum(a) / sum(gp) as apg,
	sum(tp) / sum(gp) as ppg,
	sum(perc_team_g * gp) / sum(gp) as perc_team_g,
	sum(perc_team_a * gp) / sum(gp) as perc_team_a,
	sum(perc_team_tp * gp) / sum(gp) as perc_team_tp
from skater_stats
group by 
1
,2
,3
,4
,5
having sum(gp) >= 10;

select * from team_stats where league = 'NCAA' and url = 'https://www.eliteprospects.com/team/1137/univ.-of-north-dakota/2013-2014'

select * from skater_stats where player = 'Elias Pettersson'