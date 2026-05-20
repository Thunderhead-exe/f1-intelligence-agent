# FastF1 Field Notes

LapTime: Total lap duration. It is converted to seconds for modeling.

Sector1Time, Sector2Time, Sector3Time: Timing for each of the three sectors. Sector deltas help identify localized time loss or gain.

SpeedI1 and SpeedI2: Intermediate speed measurements. Availability depends on session data.

SpeedFL: Speed at the finish line.

SpeedST: Speed trap measurement, usually useful for straight-line performance context.

Compound: Tyre compound used on the lap.

TyreLife: Approximate lap count on the tyre set.

TrackStatus: Encoded track condition information. Interpretation should be cautious because multiple statuses can occur close together.

PitInTime and PitOutTime: Timestamps for pit entry and pit exit. These identify laps that should not be treated like normal push or race laps.

Deleted: Indicates an invalidated lap. Deleted laps can be useful context but should not drive unsupported claims.

IsAccurate: FastF1 timing quality flag. Inaccurate laps should be handled carefully.

Speed: Telemetry sample speed.

RPM: Engine speed in revolutions per minute.

nGear: Gear selection.

Throttle: Throttle application percentage.

Brake: Brake signal. It can be boolean or numeric depending on data source.

DRS: DRS state. Active values can vary by encoding, so reports should describe DRS evidence carefully.

Distance: Estimated distance around the lap, used to align telemetry traces.

