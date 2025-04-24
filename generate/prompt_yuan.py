ask1 = """### Database Schemas:
# Table basketball_match, columns = [ * , teamID , schoolID , team_Name , ACC_regular_season , percentage_of_ACC , ACC_home , ACC_Street , Total_Games , percentage_of_all_games , all_home , total_street , total_neutral ]
# Table university, columns = [ * , schoolID , school , POSITION , established , association , registration , Nick_Name , basic_conference ]
# Foreign_keys = [ basketball_match.schoolID = university.schoolID ]

### Natural Language Annotations of Database Schemas:
Table basketball_match:
- Contains data related to basketball matches.
- Columns:
  - teamID: Unique identifier for each team.
  - schoolID: Identifier of the school associated with the team.
  - team_Name: Name of the team.
  - ACC_regular_season: ACC regular season statistics.
  - percentage_of_ACC: Percentage of games played in ACC.
  - ACC_home: Statistics of games played at home in ACC.
  - ACC_Street: Statistics of games played on the street in ACC.
  - Total_Games: Total number of games played.
  - percentage_of_all_games: Percentage of all games played.
  - all_home: Statistics of all games played at home.
  - total_street: Total games played on the street.
  - total_neutral: Total games played at neutral venues.

Table university:
- Stores information about universities.
- Columns:
  - schoolID: Unique identifier for each university.
  - school: Name of the university.
  - POSITION: Position of the university.
  - established: Year of establishment of the university.
  - association: Association of the university.
  - registration: Registration details of the university.
  - Nick_Name: Nickname of the university.
  - basic_conference: Basic conference affiliation of the university.

Foreign Keys:
- basketball_match.schoolID references university.schoolID, establishing a relationship between basketball matches and universities.

### Natural Language Question (NLQ): 
# Determine the cumulative count of students enrolled in colleges established after the year 1850 for each type of affiliation. Display the results in a bar chart.

### Reference DVQs:
1 - Visualize BAR SELECT Affiliation , sum(Enrollment) FROM university WHERE founded > 1850 GROUP BY affiliation
2 - Visualize BAR SELECT Affiliation , sum(Enrollment) FROM university WHERE founded > 1850 GROUP BY affiliation ORDER BY Affiliation ASC
3 - Visualize BAR SELECT Affiliation , sum(Enrollment) FROM university WHERE founded > 1850 GROUP BY affiliation ORDER BY Affiliation DESC
4 - Visualize BAR SELECT Affiliation , sum(enrollment) FROM university GROUP BY affiliation
5 - Visualize BAR SELECT Affiliation , sum(enrollment) FROM university GROUP BY affiliation ORDER BY sum(enrollment) ASC

### Possible Data Visualization Query (DVQ): 
# Visualize BAR SELECT basic_conference , SUM(Enrollment) FROM university WHERE established > 1850 GROUP BY basic_conference

#### Given Database Schemas, their corresponding Natural Language Annotations, a Natural Language Question (NLQ), please generate all possible DVQs with their probability if there are some uncertain content, like contents in "SELECT", "WHERE", "GROUP BY" or "ORDER BY". Output all possible DVQs with their probabilities as a Dictionary in JSON format. Note that the above Possible Data Visualization Query must be included in the output Possible DVQs without any modification.

#### Note: 
1. Please ensure that the generated DVQs following the syntax rules from the reference DVQs. Such as when indicating that a field is not empty, if the Reference DVQs use the form "!= \"null\"", you should also use the form "!= \"null\"", instead of "IS NOT NULL", although they are exactly the same. Note that use the double quotes to indicate the string.
2. The more certain the dvq is, the greater the probability, vice versa.
3. Do not generate other irrelevant content. Do not generate the column names that do not exist in the database schemas.
4. The probability sum of the Possible DVQs must be 1.

A: Let’s think step by step!"""


answer1="""
{
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference": 0.35,
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference ORDER BY basic_conference ASC": 0.15,
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference ORDER BY basic_conference DESC": 0.15,
    "Visualize BAR SELECT association , COUNT(*) FROM university WHERE established > 1850 GROUP BY association": 0.25,
    "Visualize BAR SELECT association , COUNT(*) FROM university WHERE established > 1850 GROUP BY association ORDER BY association ASC": 0.1
}
"""

