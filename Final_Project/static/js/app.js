$(document).ready(function() {
  // Init
  $('.image-section').hide()
  $('.loader').hide()
  $('#result').hide()

  // Upload Preview

  function readURL(input) {
    if (input.files && input.files[0]) {
      // file reader that will accept blob data or file information
      var reader = new FileReader()
      // function to file or blob data
      reader.onload = function(e) {
        $('#imagePreview').css(
          'background-image',
          'url(' + e.target.result + ')'
        )
        $('#imagePreview').hide()
        $('#imagePreview').fadeIn(650)
      }
      //   read the file data and output as URL
      reader.readAsDataURL(input.files[0])
    }
  }
  //   when the image is loaded through the bootstrap form the below will run and call the above function
  $('#imageUpload').change(function() {
    $('.image-section').show()
    $('#btn-predict').show()
    $('#result').text('')
    $('#result').hide()
    // call above function to process image to css background
    readURL(this)
  })

  // Predict
  $('#btn-predict').click(function() {
    //FormData creates key value pairs
    var form_data = new FormData($('#upload-file')[0])

    // Show loading animation
    $(this).hide()
    $('.loader').show()

    // Make prediction by calling api /predict
    $.ajax({
      type: 'POST',
      url: '/predict',
      data: form_data,
      contentType: false,
      cache: false,
      processData: false,
      async: true,
      success: function(data) {
        // Get and display the result
        $('.loader').hide()
        $('#result').fadeIn(600)
        $('#result').text(' Result:  ' + data)
        console.log('Success!')
      }
    })
  })
})

console.log('this is still working right? or is it?')
