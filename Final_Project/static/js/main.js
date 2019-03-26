$(document).ready(function() {
  'use strict'

  var window_width = $(window).width(),
    window_height = window.innerHeight,
    header_height = $('.default-header').height(),
    header_height_static = $('.site-header.static').outerHeight(),
    fitscreen = window_height - header_height

  $('.fullscreen').css('height', window_height)
  $('.fitscreen').css('height', fitscreen)

  //-------- Active Sticky Js ----------//
  $('.default-header').sticky({ topSpacing: 0 })

  //------- Active Nice Select --------//
  $('select').niceSelect()

  // -------   Active Mobile Menu-----//

  $('.menu-bar').on('click', function(e) {
    e.preventDefault()
    $('nav').toggleClass('hide')
    $('span', this).toggleClass('lnr-menu lnr-cross')
    $('.main-menu').addClass('mobile-menu')
  })

  $('.nav-item a:first').tab('show')

  // Select all links with hashes
  $('.main-menubar a[href*="#"]')
    // Remove links that don't actually link to anything
    .not('[href="#"]')
    .not('[href="#0"]')
    .click(function(event) {
      // On-page links
      if (
        location.pathname.replace(/^\//, '') ==
          this.pathname.replace(/^\//, '') &&
        location.hostname == this.hostname
      ) {
        // Figure out element to scroll to
        var target = $(this.hash)
        target = target.length ? target : $('[name=' + this.hash.slice(1) + ']')
        // Does a scroll target exist?
        if (target.length) {
          // Only prevent default if animation is actually gonna happen
          event.preventDefault()
          $('html, body').animate(
            {
              scrollTop: target.offset().top - 68
            },
            1000,
            function() {
              // Callback after animation
              // Must change focus!
              var $target = $(target)
              $target.focus()
              if ($target.is(':focus')) {
                // Checking if the target was focused
                return false
              } else {
                $target.attr('tabindex', '-1') // Adding tabindex for elements not focusable
                $target.focus() // Set focus again
              }
            }
          )
        }
      }
    })

  //  Counter Js

  $('.counter').counterUp({
    delay: 10,
    time: 1000
  })

  // -------   Mail Send ajax

  $(document).ready(function() {
    var form = $('#myForm') // contact form
    var submit = $('.submit-btn') // submit button
    var alert = $('.alert-msg') // alert div for show alert message

    // form submit event
    form.on('submit', function(e) {
      e.preventDefault() // prevent default form submit

      $.ajax({
        url: 'mail.php', // form action url
        type: 'POST', // form submit method get/post
        dataType: 'html', // request type html/json/xml
        data: form.serialize(), // serialize form data
        beforeSend: function() {
          alert.fadeOut()
          submit.html('Sending....') // change submit button text
        },
        success: function(data) {
          alert.html(data).fadeIn() // fade in response data
          form.trigger('reset') // reset form
          submit.attr('style', 'display: none !important') // reset submit button text
        },
        error: function(e) {
          console.log(e)
        }
      })
    })
  })

  // ------- Image load function

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
          // $('.loader').hide()
          // $('#result').fadeIn(600)
          // $('#result').text(' Result:  ' + data)
          // console.log(data)
          console.log('Success!')
        }
      })
    })
  })
})

// d3 to grab route data and append p tag

setTimeout(function() {
  description_tag = d3.select('#btn-predict')

  description_tag.on('click', function() {
    url = '/predict'

    d3.json(url, function(obj) {
      console.log(obj)

      first_name = obj[0][0]
      first_name_val = obj[0][1]

      second_name = obj[1][0]
      second_name_val = obj[1][1]

      third_name = obj[2][0]
      third_name_val = obj[2][1]

      console.log(
        `Your diagnosis: There is a ${first_name_val} you may have ${first_name}, a ${second_name_val} you have ${first_name}, and ${third_name_val} it's ${third_name}`
      )

      function removeDiv() {
        d3.select('#two').remove()
        d3.select('#one')
          .append('p')
          .attr('id', 'myNewParagrap')
          .append('text')
          .text(
            `Result: ${first_name_val} probabilty you have ${first_name}, a ${second_name_val} you have ${first_name}, and ${third_name_val} it's ${third_name}.`
          )
      }
      removeDiv()
    })
  }),
    30000000
})
