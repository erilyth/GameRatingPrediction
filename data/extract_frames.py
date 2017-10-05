import csv
import glob
import os
import os.path
from subprocess import call

def extract_files():
    """
    Extract frames from video in train/test folders with class directories

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    folders = ['./train/', './test/']

    for folder in folders:
        class_folders = glob.glob(folder + '*')

        for vid_class in class_folders:
            class_files = glob.glob(vid_class + '/*.avi')

            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_video_parts(video_path)

                train_or_test, classname, filename_no_ext, filename = video_parts

                if not check_already_extracted(video_parts):
                    src = train_or_test + '/' + classname + '/' + \
                        filename
                    dest = train_or_test + '/' + classname + '/' + \
                        filename_no_ext + '-%04d.jpg'
                    call(["ffmpeg", "-i", src, dest])

                nb_frames = get_nb_frames_for_video(video_parts)

                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    # Get number of frames in video
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(train_or_test + '/' + classname + '/' +
                                filename_no_ext + '*.jpg')
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split('/')
    filename = parts[3]
    filename_no_ext = filename.split('.')[0]
    classname = parts[2]
    train_or_test = parts[1]

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(train_or_test + '/' + classname +
                               '/' + filename_no_ext + '-0001.jpg'))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. 
    It will have format: [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()
