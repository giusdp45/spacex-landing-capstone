-- 01_unique_launch_sites
SELECT DISTINCT LaunchSite FROM launches ORDER BY LaunchSite;

-- 02_ccafs_first_five
SELECT Date, FlightNumber, LaunchSite FROM launches WHERE LaunchSite LIKE 'CCAFS%' ORDER BY Date LIMIT 5;

-- 03_nasa_crs_total_payload
SELECT COUNT(*) AS Launches, COUNT(PayloadMass) AS KnownMasses, SUM(PayloadMass) AS KnownPayloadKg FROM launches WHERE Customer LIKE '%NASA%CRS%';

-- 04_f9_v1_1_mean_payload
SELECT COUNT(*) AS Launches, COUNT(PayloadMass) AS KnownMasses, AVG(PayloadMass) AS MeanPayloadKg FROM launches WHERE BoosterVersion LIKE 'F9 v1.1%';

-- 05_first_successful_ground_landing
SELECT MIN(Date) AS FirstSuccessfulGroundLanding FROM launches WHERE LandingSuccess=1 AND LandingType='Ground pad';

-- 06_drone_landings_payload_4000_6000
SELECT Date, BoosterVersion, PayloadMass FROM launches WHERE LandingSuccess=1 AND LandingType='Drone ship' AND PayloadMass>4000 AND PayloadMass<6000 ORDER BY Date;

-- 07_launch_outcome_counts
SELECT LaunchSuccess, COUNT(*) AS Launches FROM launches GROUP BY LaunchSuccess;

-- 08_max_payload_boosters
SELECT Date, BoosterVersion, PayloadMass FROM launches WHERE PayloadMass=(SELECT MAX(PayloadMass) FROM launches) ORDER BY Date;

-- 09_failed_drone_landings_2015
SELECT strftime('%m',Date) AS Month, BoosterVersion, LaunchSite, LandingOutcome FROM launches WHERE LandingResult='Failure' AND LandingType='Drone ship' AND strftime('%Y',Date)='2015' ORDER BY Date;

-- 10_landing_outcomes_ranked
SELECT LandingResult, LandingType, COUNT(*) AS Launches FROM launches WHERE Date BETWEEN '2010-06-04' AND '2017-03-20' GROUP BY LandingResult, LandingType ORDER BY Launches DESC;

-- 11_site_landing_success_rates
SELECT LaunchSite, COUNT(*) AS Launches, SUM(LandingSuccess) AS SuccessfulLandings, ROUND(100.0*AVG(LandingSuccess),1) AS LandingSuccessPct FROM launches GROUP BY LaunchSite ORDER BY LandingSuccessPct DESC;