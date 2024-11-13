Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });
    
    dz.on("addedfile", function() {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    dz.on("complete", function (file) {
        if (!file.type.match(/image.*/)) {
            $("#error").text("File is not an image. Please upload a valid image file.");
            console.log("File is not an image.");
            alert("File is not an image. Please upload a valid image file.")
            $("#resultHolder").hide();
            $("#divClassTable").hide();                
            $("#error").show();
            return;
        }
        
        let imageData = file.dataURL;
        console.log("File data received:");

        var url = "http://127.0.0.1:5000/classify_image";
        $.post(url, {
            image_data: file.dataURL
        }, function(data, status) {
            console.log("Data received:", data);
            if (!data || data.length == 0) {
                $("#error").text("Can't classify image. Classifier was not able to detect face and two eyes properly.");
                console.log("No data received");
                alert("Can't classify image. Classifier was not able to detect face and two eyes properly.")
                $("#resultHolder").hide();
                $("#divClassTable").hide();                
                $("#error").show();
                return;
            }

            let match = null;
            let bestScore = -1;
            for (let i = 0; i < data.length; ++i) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                if (maxScoreForThisClass > bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }

            // if(bestScore < 70){
            //     $("#error").text("Can't classify image.");
            //     alert("Celebrity Not found")
            //     $("#resultHolder").hide();
            //     $("#divClassTable").hide();                
            //     $("#error").show();
            //     return;
            // }

            // Reset all rows to default color
            $("#classTable tbody tr").each(function() {
                $(this).find('td').css('color', '#ffffff'); // Default color
            });

            if (data) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();
                $("#resultHolder").html($(`[data-player="${match.class}"]`).html());

                let classDictionary = match.class_dictionary;
                for (let personName in classDictionary) {
                    let index = classDictionary[personName];
                    let probabilityScore = match.class_probability[index];
                    let scoreElement = `#score_${personName}`; // Get the probability score cell
                    let nameElement = `#name_${personName}`; // Get the probability score cell
                    
                    $(scoreElement).html(probabilityScore);
                    
                    // Highlight matched person's name and score in green
                    if (personName === match.class) {
                        $(scoreElement).css('color', '#05cf05', 'font-weight', 'bold');
                        $(nameElement).css('color', '#05cf05', 'font-weight', 'bold');
                    }
                }
            }

            // dz.removeFile(file);            
        });
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();		
    });
}

$(document).ready(function() {
    console.log("ready!");
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});



/* 
            Below is a sample response if you have two faces in an image lets say virat and roger together.
            Most of the time if there is one person in the image you will get only one element in below array
            data = [
                {
                    class: "viral_kohli",
                    class_probability: [1.05, 12.67, 22.00, 4.5, 91.56],
                    class_dictionary: {
                        lionel_messi: 0,
                        maria_sharapova: 1,
                        roger_federer: 2,
                        serena_williams: 3,
                        virat_kohli: 4
                    }
                },
                {
                    class: "roder_federer",
                    class_probability: [7.02, 23.7, 52.00, 6.1, 1.62],
                    class_dictionary: {
                        lionel_messi: 0,
                        maria_sharapova: 1,
                        roger_federer: 2,
                        serena_williams: 3,
                        virat_kohli: 4
                    }
                }
            ]
            */