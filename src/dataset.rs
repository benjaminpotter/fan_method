use chrono::{DateTime, TimeZone, Utc};
use rumpus::image::{IntensityImage, RayImage};
use std::{error::Error, path::Path};

pub fn read_time<P: AsRef<Path>>(path: P) -> Result<Vec<DateTime<Utc>>, Box<dyn Error + 'static>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut frames = Vec::new();
    for result in reader.records() {
        let record = result?;

        let start_idx = 17;
        let year: i32 = record.get(start_idx + 0).unwrap().parse()?;
        assert_eq!(year, 2025);
        let month: u32 = record.get(start_idx + 1).unwrap().parse()?;
        let day: u32 = record.get(start_idx + 2).unwrap().parse()?;
        let hour: u32 = record.get(start_idx + 3).unwrap().parse()?;
        let min: u32 = record.get(start_idx + 4).unwrap().parse()?;
        let msec: u32 = record.get(start_idx + 5).unwrap().parse()?;
        let sec = msec / 1000;

        let time = Utc
            .with_ymd_and_hms(year, month, day, hour, min, sec)
            .unwrap();
        frames.push(time);
    }

    Ok(frames)
}

#[derive(Debug)]
pub struct InsFrame {
    pub lat: f64,
    pub lon: f64,
    // Clockwise from north
    pub azimuth: f64,
    pub pitch: f64,
    pub roll: f64,
}

pub fn read_ins<P: AsRef<Path>>(path: P) -> Result<Vec<InsFrame>, Box<dyn Error + 'static>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut frames = Vec::new();
    for result in reader.records() {
        let record = result?;

        let lat = record.get(13).unwrap().parse()?;
        let lon = record.get(14).unwrap().parse()?;

        let roll = record.get(19).unwrap().parse()?;
        let pitch = record.get(20).unwrap().parse()?;
        let azimuth = record.get(21).unwrap().parse()?;

        frames.push(InsFrame {
            lat,
            lon,
            azimuth,
            pitch,
            roll,
        });
    }

    Ok(frames)
}

pub fn read_image<P: AsRef<Path>>(path: P) -> Result<Vec<f64>, Box<dyn Error + 'static>> {
    // Open a new image and ensure it is in single channel greyscale format.
    let raw_image = image::ImageReader::open(&path)?.decode()?.into_luma8();

    // Create a new IntensityImage from the input image.
    let (width, height) = raw_image.dimensions();
    let intensity_image =
        IntensityImage::from_bytes(width as usize, height as usize, &raw_image.into_raw())
            .expect("image dimensions are even");

    Ok(RayImage::from_metapixels(
        intensity_image.metapixels(),
        intensity_image.rows(),
        intensity_image.cols(),
    )?
    .rays()
    .filter_map(|ray| Some(ray?.aop().as_radians()))
    .collect())
}
